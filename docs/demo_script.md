# 演示脚本

## 1. 开场介绍（30 秒）

这是一个基于 LangGraph StateGraph 编排的深度研究助手，输入研究主题后，系统会自动完成：

1. 主题拆解
2. Web 候选搜索与知识库检索
3. 证据整理
4. ReAct Writer 带引用写作
5. 自动审查
6. 输出报告与 Trace

核心亮点不是多智能体数量，而是：**可追溯、可观测、可量化**。

## 2. 演示命令

### 配置检查

```bash
conda activate researchcopilot
python main.py --check-config
```

### CLI 演示

```bash
python main.py --run-topic "RAG 系统中 Hybrid Retrieval 的价值"
```

### 单独调试 Web 检索

```bash
python scripts/debug_web_search.py --query "OpenAI GPT-5.4 的定位与能力变化" --max-results 10
```

### Streamlit 演示

```bash
streamlit run app.py
```

## 3. 建议展示主题

- 2025 年大模型推理优化技术综述
- RAG 系统中 Hybrid Retrieval 的价值
- Agent Workflow 和 Autonomous Agent 的取舍

## 4. 演示重点

### 页面中重点看哪里

- 子问题拆解结果
- 每个子问题的 Query History
- Web 搜索候选数、LLM 筛选后 URL 数、fetch 成功数
- 证据列表和 evidence_id
- Writer 节点如何通过工具读取分节证据
- 最终带引用报告
- Reviewer 评分结果
- 审查失败后重新规划与补检索的闭环
- Trace 日志
- 实验结果概览

### 可以主动讲的点

- 为什么采用 workflow 而不是完全自治多智能体
- 为什么要做 Hybrid Retrieval 和 Re-rank
- 为什么要做 Query Rewrite 和重试
- 为什么要保存运行工件和 Trace

## 5. 结尾话术

这个项目的重点不是把 API 串起来，而是把研究型 Agent 系统做成一个**能跑、能看、能测、能解释**的工程原型。
