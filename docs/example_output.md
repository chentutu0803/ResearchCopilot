# 示例输出说明

本项目的真实运行输出会保存在：

- `experiments/results/latest_run.json`
- `experiments/results/*_run.json`

你可以直接展示这些文件中的：

- `topic`
- `sub_questions`
- `sections`
- `final_report`
- `trace`
- `run_summary`

## 推荐展示方式

1. 先展示 Streamlit 页面中的最终报告
2. 再展示 `latest_run.json`，说明系统如何保存运行工件
3. 最后展示 `retrieval_eval_summary.json` 和 `workflow_ablation_summary.json`
4. 补充展示 `docs/screenshots/` 里的演示截图素材

## 面试时可强调

- 最终报告不是黑盒输出，而是有完整中间证据链
- 每次运行都会留下结构化工件，便于调试和复盘
- 评测和消融实验可以帮助分析系统设计取舍

## 截图资产

执行下列命令后，会生成演示截图：

```bash
python experiments/generate_demo_assets.py
```

输出目录：`docs/screenshots/`
