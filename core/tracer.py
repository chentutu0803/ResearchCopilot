from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.schemas import TraceEvent


class TraceCollector:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def add_event(
        self,
        *,
        event_type: str,
        step_name: str,
        status: str,
        input_summary: str = "",
        output_summary: str = "",
        tool_name: str | None = None,
        duration_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_type=event_type,
            step_name=step_name,
            status=status,
            input_summary=input_summary,
            output_summary=output_summary,
            tool_name=tool_name,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    def clear(self) -> None:
        self._events.clear()

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self._events]

    def save_json(self, file_path: str | Path) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict_list(), file, ensure_ascii=False, indent=2)
