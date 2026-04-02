from __future__ import annotations

import argparse

from core.langgraph_workflow import LangGraphResearchWorkflow
from core.settings import get_settings
from core.tracer import TraceCollector


def run_check_config() -> None:
    settings = get_settings()
    tracer = TraceCollector()

    settings.trace_path.mkdir(parents=True, exist_ok=True)
    settings.data_path.mkdir(parents=True, exist_ok=True)

    tracer.add_event(
        event_type="system",
        step_name="startup.check_config",
        status="success",
        input_summary="执行基础配置检查",
        output_summary="配置读取成功",
        metadata={
            "app_name": settings.app_name,
            "app_env": settings.app_env,
            "model_name": settings.model_name,
            "has_llm_credentials": settings.has_llm_credentials,
        },
    )

    trace_file = settings.trace_path / "startup_trace.json"
    tracer.save_json(trace_file)

    print("配置检查完成")
    print(f"项目目录: {settings.project_root}")
    print(f"数据目录: {settings.data_path}")
    print(f"Trace 目录: {settings.trace_path}")
    print(f"模型名称: {settings.model_name}")
    print(f"是否已配置 API Key: {settings.has_llm_credentials}")
    print(f"启动日志已写入: {trace_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ResearchCopilot 基础入口")
    parser.add_argument(
        "--run-topic",
        type=str,
        default="",
        help="直接在命令行执行一个研究主题",
    )
    parser.add_argument(
        "--report-language",
        type=str,
        default="",
        help="报告语言，例如：中文 或 English",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="检查配置是否可读取，并写入一条启动 Trace",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.check_config:
        run_check_config()
        return

    if args.run_topic:
        settings = get_settings()
        workflow = LangGraphResearchWorkflow(settings=settings)
        result = workflow.run(
            topic=args.run_topic,
            use_web=False,
            use_kb=True,
            report_language=args.report_language or settings.report_language,
        )
        print(result.final_report.content)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
