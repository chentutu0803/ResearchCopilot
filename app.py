from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from core.langgraph_workflow import LangGraphResearchWorkflow
from core.settings import get_settings


def _load_json(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


st.set_page_config(page_title="研究副驾驶", page_icon="🔬", layout="wide")

settings = get_settings()

st.title("🔬 研究副驾驶（ResearchCopilot）")
st.caption("一个可验证、可追溯、可观测的深度研究助手")

if settings.has_llm_credentials:
    st.success("当前处于 LLM 模式：已检测到 DeepSeek API Key。")
else:
    st.warning(
        "当前处于启发式模式：未检测到 DeepSeek API Key，将使用模板规划与模板写作。"
    )

with st.sidebar:
    st.subheader("运行配置")
    use_web = st.checkbox("启用 Web Search", value=False)
    use_kb = st.checkbox("启用本地知识库", value=True)
    report_language = st.selectbox("报告语言", options=["中文", "English"], index=0)
    enable_reviewer = st.checkbox("启用 Reviewer", value=True)
    allow_query_rewrite = st.checkbox("启用 Query Rewrite", value=True)
    max_subquestions = st.slider("子问题上限（0=自动）", min_value=0, max_value=6, value=0)
    max_search_rounds = st.slider("最大搜索轮数", min_value=1, max_value=5, value=3)
    st.markdown("---")
    st.subheader("当前实现特性")
    st.write("- 主题规划")
    st.write("- Web Search")
    st.write("- 本地轻量混合检索")
    st.write("- 带引用写作")
    st.write("- 自动审查")
    st.write("- 全流程 Trace")

topic = st.text_input(
    "请输入研究主题", placeholder="例如：2025 年大模型推理优化技术综述"
)

if st.button("开始研究"):
    if not topic.strip():
        st.error("请输入研究主题后再运行。")
    else:
        workflow = LangGraphResearchWorkflow(settings=settings)
        progress_box = st.status("执行中", expanded=True)
        with st.spinner("正在执行研究工作流，请稍候..."):
            result = workflow.run(
                topic=topic,
                use_web=use_web,
                use_kb=use_kb,
                report_language=report_language,
                enable_reviewer=enable_reviewer,
                allow_query_rewrite=allow_query_rewrite,
                max_subquestions=max_subquestions,
                max_search_rounds=max_search_rounds,
                progress_callback=progress_box.write,
            )
        progress_box.update(label="执行完成", state="complete", expanded=True)

        st.subheader("研究报告")
        st.markdown(result.final_report.content)

        st.subheader("子问题")
        for item in result.sub_questions:
            st.markdown(f"- {item.question}")

        st.subheader("研究分节")
        for section in result.sections:
            with st.expander(section.sub_question):
                st.markdown("**检索摘要**")
                st.text(section.summary)
                st.markdown("**Query 历史**")
                st.write(section.query_history)
                st.markdown("**证据列表**")
                for evidence in section.evidence_items[:8]:
                    st.markdown(
                        f"- `{evidence.evidence_id}` **{evidence.title}**  \\n来源：{evidence.source_url or '本地知识库'}"
                    )

        st.subheader("审查结果")
        review = result.final_report.review
        if review is not None:
            st.write(
                {
                    "accepted": review.accepted,
                    "factual_support": review.score.factual_support,
                    "citation_coverage": review.score.citation_coverage,
                    "coherence": review.score.coherence,
                    "completeness": review.score.completeness,
                    "total": review.score.total,
                }
            )
            if review.suggestions:
                st.markdown("**审查反馈**")
                for suggestion in review.suggestions:
                    st.markdown(f"- {suggestion}")

        st.subheader("实验结果概览")
        eval_summary = _load_json(settings.trace_path / "retrieval_eval_summary.json")
        ablation_summary = _load_json(
            settings.trace_path / "workflow_ablation_summary.json"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**检索评测**")
            st.json(eval_summary if eval_summary else {"status": "尚未生成评测摘要"})
        with col2:
            st.markdown("**工作流消融**")
            st.json(
                ablation_summary if ablation_summary else {"status": "尚未生成消融摘要"}
            )

        with st.sidebar:
            st.subheader("Trace 日志")
            st.dataframe(
                result.model_dump(mode="json")["trace"], use_container_width=True
            )

st.info("建议先关闭 Web Search，用本地知识库验证完整链路；确认可跑后再打开网络搜索。")
