from __future__ import annotations

import json
import sys
import time
from typing import Any

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from core.exceptions import ConfigurationError, LLMClientError
from core.settings import Settings
from core.tracer import TraceCollector


class LLMClient:
    def __init__(
        self, settings: Settings, tracer: TraceCollector | None = None
    ) -> None:
        self.settings = settings
        self.tracer = tracer

        if not settings.has_llm_credentials:
            raise ConfigurationError(
                "未检测到 DEEPSEEK_API_KEY，无法初始化大模型客户端。"
            )

        client_kwargs: dict[str, str] = {"api_key": settings.deepseek_api_key}
        if settings.deepseek_base_url.strip():
            client_kwargs["base_url"] = settings.deepseek_base_url.strip()

        self.client = OpenAI(**client_kwargs)

    def _debug(self, message: str) -> None:
        if self.settings.log_level.upper() != "DEBUG":
            return
        timestamp = time.strftime("%H:%M:%S")
        print(f"[DEBUG llm_client {timestamp}] {message}", file=sys.stderr, flush=True)

    def generate_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        started_at = time.perf_counter()
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            )
        messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

        response = None
        content_preview = ""
        last_exception: Exception | None = None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                self._debug(
                    f"generate_text.start attempt={attempt + 1} "
                    f"prompt_len={len(prompt)} max_tokens={max_tokens} "
                    f"temperature={temperature}"
                )
                stream = self.client.chat.completions.create(
                    model=self.settings.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.settings.llm_timeout_seconds,
                    stream=True,
                )
                chunks: list[str] = []
                response = None
                for chunk in stream:
                    response = chunk
                    if not chunk.choices:
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    piece = getattr(delta, "content", None) if delta is not None else None
                    if piece:
                        chunks.append(piece)
                content_preview = "".join(chunks)
                self._debug(
                    f"generate_text.done attempt={attempt + 1} "
                    f"content_len={len(content_preview)}"
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                self._debug(
                    f"generate_text.error attempt={attempt + 1} error={type(exc).__name__}: {exc}"
                )
                if attempt >= self.settings.llm_max_retries:
                    duration_ms = int((time.perf_counter() - started_at) * 1000)
                    if self.tracer is not None:
                        self.tracer.add_event(
                            event_type="llm",
                            step_name="llm.generate_text",
                            status="error",
                            input_summary=prompt[:200],
                            output_summary=str(exc),
                            duration_ms=duration_ms,
                            metadata={"attempt": attempt + 1},
                        )
                    raise LLMClientError(f"模型调用失败: {exc}") from exc
                time.sleep(1.2 * (attempt + 1))

        if response is None and last_exception is not None:
            raise LLMClientError(f"模型调用失败: {last_exception}") from last_exception
        if response is None:
            raise LLMClientError("模型调用失败：未获得响应。")
        assert response is not None

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        content = content_preview
        usage = getattr(response, "usage", None)

        if self.tracer is not None:
            self.tracer.add_event(
                event_type="llm",
                step_name="llm.generate_text",
                status="success",
                input_summary=prompt[:200],
                output_summary=content[:200],
                duration_ms=duration_ms,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                metadata={
                    "model_name": self.settings.model_name,
                    "timeout_seconds": self.settings.llm_timeout_seconds,
                },
            )

        return content

    def generate_json(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        instruction = (
            "请严格返回 JSON 对象，不要输出 Markdown 代码块，不要输出额外解释。\n"
            f"用户任务如下：\n{prompt}"
        )
        raw_text = self.generate_text(
            prompt=instruction,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"模型输出不是合法 JSON: {raw_text}") from exc
