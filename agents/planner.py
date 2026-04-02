from __future__ import annotations

from core.llm_client import LLMClient
from core.schemas import SubQuestion
from core.tracer import TraceCollector


class PlannerAgent:
    def __init__(
        self, llm_client: LLMClient | None = None, tracer: TraceCollector | None = None
    ) -> None:
        self.llm_client = llm_client
        self.tracer = tracer

    def plan(
        self,
        topic: str,
        max_subquestions: int = 0,
        review_feedback: list[str] | None = None,
        previous_subquestions: list[SubQuestion] | None = None,
    ) -> list[SubQuestion]:
        questions = (
            self._plan_with_llm(
                topic,
                max_subquestions,
                review_feedback=review_feedback,
                previous_subquestions=previous_subquestions,
            )
            if self.llm_client
            else []
        )
        if not questions:
            questions = self._plan_with_template(
                topic,
                max_subquestions,
                review_feedback=review_feedback,
            )

        if self.tracer is not None:
            self.tracer.add_event(
                event_type="agent",
                step_name="agent.planner",
                status="success",
                input_summary=topic,
                output_summary=f"生成 {len(questions)} 个子问题",
                metadata={"review_feedback": review_feedback or []},
            )
        return questions

    def _plan_with_llm(
        self,
        topic: str,
        max_subquestions: int,
        review_feedback: list[str] | None = None,
        previous_subquestions: list[SubQuestion] | None = None,
    ) -> list[SubQuestion]:
        assert self.llm_client is not None
        previous_questions = previous_subquestions or []
        feedback_block = (
            "\n".join(f"- {item}" for item in (review_feedback or [])) or "无"
        )
        previous_block = (
            "\n".join(f"- {item.question}" for item in previous_questions) or "无"
        )
        prompt = f"""
请把研究主题拆成若干个彼此不重复、适合检索和写作的子问题。
对于复杂技术问题，优先拆成 3-6 个子问题；只有在主题明显很简单时，才少于 3 个。
如果调用方给了正整数上限，则不要超过该上限；如果上限为 0，则由你根据主题复杂度自行决定。
如果已有审查反馈，请优先根据反馈补足证据不足、结构不完整或引用不足的部分。
返回 JSON，格式如下：
{{
  "sub_questions": [
    {{"question": "...", "rationale": "..."}}
  ]
}}

研究主题：{topic}
上一轮子问题：
{previous_block}

审查反馈：
{feedback_block}
""".strip()

        try:
            data = self.llm_client.generate_json(prompt=prompt, temperature=0.1)
        except Exception:  # noqa: BLE001
            return []

        raw_items = data.get("sub_questions", [])
        questions: list[SubQuestion] = []
        limited_items = (
            raw_items[:max_subquestions]
            if max_subquestions and max_subquestions > 0
            else raw_items
        )
        for index, item in enumerate(limited_items, start=1):
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            questions.append(
                SubQuestion(
                    index=index,
                    question=question,
                    rationale=str(item.get("rationale", "")).strip(),
                )
            )
        return questions

    def _plan_with_template(
        self,
        topic: str,
        max_subquestions: int,
        review_feedback: list[str] | None = None,
    ) -> list[SubQuestion]:
        templates: list[str] = []
        if review_feedback:
            templates.extend(
                [
                    "{topic} 里哪些关键结论最需要补充证据和引用？",
                    "{topic} 里哪些内容最容易因为资料不足而写得不完整？",
                ]
            )
        templates.extend(
            [
                "{topic} 的背景、定义和核心问题是什么？",
                "{topic} 的关键技术路线或核心方法有哪些？",
                "{topic} 当前的工程优化重点和实践难点是什么？",
                "{topic} 的典型应用场景、优势与局限是什么？",
                "{topic} 的发展趋势和未来值得关注的方向是什么？",
            ]
        )
        questions = [
            SubQuestion(
                index=index, question=template.format(topic=topic), rationale="模板规划"
            )
            for index, template in enumerate(
                templates[:max_subquestions] if max_subquestions and max_subquestions > 0 else templates,
                start=1,
            )
        ]
        return questions
