# 🔬 研究副驾驶（ResearchCopilot）

一个**可验证、可追溯、可观测**的 AI 深度研究助手。  
输入一个研究主题，系统自动完成：**问题拆解 → 搜索与检索 → 证据整理 → 带引用写作 → 自动审查 → 输出研究报告与 Trace**。

> 这个项目的重点不是“堆多少个 Agent”，而是把研究型 Agent 系统做成一个 **能跑、能看、能测、能解释** 的工程原型。

---

## 一、项目适合展示什么能力

这个项目适合用于面试以下方向：

- 大模型开发
- Agent 应用开发
- RAG / 检索增强生成工程岗位
- 智能体工作流与工具调用方向岗位

它重点体现的不是“调 API”，而是以下几类能力：

1. **工作流设计能力**：为什么用 workflow，而不是完全自治多智能体
2. **RAG 工程能力**：Hybrid Retrieval、精排、引用约束
3. **工具调用能力**：Web Search、知识库检索、Query Rewrite
4. **系统调试能力**：Trace、结构化运行工件、实验摘要
5. **评测意识**：检索评测、工作流消融、指标输出

---

## 二、项目核心亮点

### 1. 带引用的研究报告生成
- 每个关键结论都能映射到具体证据项
- 输出包含引用编号和参考资料列表
- 可直接向面试官展示“不是黑盒生成，而是有证据链支撑”

### 2. 标准 RAG 检索链路
- BM25：处理关键词、缩写词、专有名词
- Qdrant 官方 BM25：支持更标准的文本检索和 `multilingual tokenizer`
- Qdrant Local：作为本地向量数据库持久化 dense vector
- Sentence-Transformers：生成 dense embedding，处理语义相似表达
- Re-rank：提升前几条结果相关性
- 离线建库：先构建向量库，查询阶段只做检索，不自动重建

### 3. 多阶段 Web Research 检索
- Tavily Search：召回 20 条左右 web 候选结果
- 规则预筛选：去重并过滤明显不可用站点，对 `release / docs / README / repo root` 等路径加分
- LLM Rerank：基于 `title + url + snippet` 删除无关结果并排序，而不是直接按搜索引擎顺序抓取
- Web Fetch：只对筛选后的页面抓正文，并把成功抓取的全文证据一次性交给 Writer

### 4. Query Rewrite 与失败重试
- 搜索为空或质量不佳时，不直接结束
- 自动进行 query 改写和继续尝试
- 更贴近真实工程系统，而不是一次失败就崩

### 5. Reviewer 有明确边界
- 不是无限循环调模型
- 有评分维度、有阈值、有闭环次数上限
- 这一点很适合回答“你这个 Critic 有什么实际价值”

### 6. 全流程可观测
- 每一步都有 Trace
- 每次运行都会保存结构化工件
- 可以定位是规划问题、检索问题还是写作问题
- Web 检索链路会保存 `raw_candidates / filtered_candidates / selected_candidates / fetch_details`
- Writer ReAct 节点会保存完整 `writer_trace`
- `run_summary` 会直接汇总 `web_search_round_count / writer_react_turn_count / reviewer_replan_count`

### 7. 有评测，不只是 Demo
- 检索评测：Hit@3、MRR@5
- 工作流消融：搜索轮数、关闭 Query Rewrite、关闭 Reviewer
- 工作流指标：引用覆盖率、不支持结论比例、平均搜索轮数等

---

## 三、系统架构

整体流程：

`用户主题 -> LangGraph StateGraph -> Planner -> Researcher -> ReAct Writer -> Reviewer -> Final Report + Trace + Metrics`

### 模块说明

- **LangGraph StateGraph**：负责编排规划、检索、写作、审查和回退逻辑
- **Planner**：将主题拆分为 3-6 个可研究子问题
- **Researcher**：执行受限检索循环，调用 Web Search / Knowledge Base
- **ReAct Writer**：通过工具读取子问题和分节证据，再生成带引用研究报告
- **Reviewer**：按子问题覆盖率、证据充分性、来源质量和引用合法性评审，不通过时触发重新规划与补检索
- **Tracer**：记录全过程执行日志
- **Evaluator**：输出检索评测和工作流指标

---

## 四、当前已实现内容

### 核心工作流
- [x] Planner / Researcher / Writer / Reviewer 完整链路
- [x] CLI 入口
- [x] Streamlit 演示页面
- [x] 运行进度展示

### 工具与检索
- [x] Tavily Search
- [x] 本地知识库检索
- [x] Qdrant Local 向量库
- [x] Qdrant 官方 BM25 / multilingual tokenizer
- [x] Sentence-Transformers 文本向量化
- [x] Web 候选结果规则预筛选
- [x] Web 候选结果 LLM Rerank
- [x] 离线建库脚本
- [x] Hybrid Retrieval
- [x] Re-rank
- [x] Citation Checker

### 可观测与评测
- [x] Trace 日志
- [x] 运行工件落盘
- [x] 检索评测脚本
- [x] 工作流消融脚本
- [x] Evaluator 指标脚本

### 面试材料
- [x] 实施计划文档
- [x] 演示脚本
- [x] 示例输出说明
- [x] 演示截图素材

---

## 五、目录结构

```text
ResearchCopilot/
├── implementation_plan.md
├── README.md
├── requirements.txt
├── requirements-rag.txt
├── .env.example
├── app.py
├── config.py
├── main.py
│
├── core/
├── agents/
├── tools/
├── rag/
├── data/
├── docs/
├── experiments/
└── tests/
```

---

## 六、快速开始

### 1. 创建环境并安装依赖

```bash
conda create -n researchcopilot python=3.12 -y
conda activate researchcopilot
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

如果要启用标准 RAG 检索链路，再额外安装：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-rag.txt
```

Web Search 现在走 Tavily：

- 需要配置 `TAVILY_API_KEY`
- 搜索阶段会先召回候选结果，再做规则预筛选与 LLM 排序
- 只有筛选后的页面才会进入正文抓取阶段

### 2. 配置环境变量

复制 `.env.example` 为 `.env`：

```env
DEEPSEEK_API_KEY=
MODEL_NAME=deepseek-chat
DEEPSEEK_BASE_URL=
TAVILY_API_KEY=
KB_BACKEND=qdrant
QDRANT_PATH=./data/qdrant
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_CLOUD_INFERENCE=true
QDRANT_BM25_TOKENIZER=multilingual
EMBEDDING_BACKEND=sentence_transformer
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
HF_ENDPOINT=https://hf-mirror.com
WEB_FETCH_PAGES=1
```

### 3. 构建知识库

先离线构建向量库：

```bash
python scripts/build_kb.py
```

如果你启用了 `QDRANT_URL + QDRANT_API_KEY + QDRANT_CLOUD_INFERENCE=true`，构建时会同时走 Qdrant 官方 `BM25 + multilingual tokenizer` 这条路。

如果你更新了 `data/raw/` 下的知识库文档，需要显式重建：

```bash
python scripts/rebuild_kb.py
```

查询路径不会自动重建向量库；如果索引不存在或文档已更新但未重建，程序会直接报错并提示先执行建库脚本。

### 4. 基础检查

```bash
python main.py --check-config
```

### 5. CLI 演示

```bash
python main.py --run-topic "RAG 系统中 Hybrid Retrieval 的价值"
```

如果你想单独调试 Web 检索链路：

```bash
python scripts/debug_web_search.py --query "OpenAI GPT-5.4 的定位与能力变化" --max-results 10
```

### 6. 启动页面

```bash
streamlit run app.py
```

---

## 七、评测与实验

### 1. 检索评测

```bash
python experiments/retrieval_eval.py
```

输出摘要：

- `experiments/results/retrieval_eval_summary.json`

当前评测包含：

- `Hit@3`
- `MRR@5`
- 检索方法对比：`bm25 / vector / hybrid`

### 2. 工作流消融

```bash
python experiments/workflow_ablation.py
```

输出摘要：

- `experiments/results/workflow_ablation_summary.json`

当前对比包含：

- 搜索轮数变化
- 关闭 Query Rewrite
- 关闭 Reviewer

### 3. 工作流指标

```bash
python experiments/evaluator.py
```

输出摘要：

- `experiments/results/workflow_metrics_summary.json`

当前指标包含：

- `citation_coverage_rate`
- `unsupported_claim_rate`
- `reviewer_trigger_rate`
- `average_search_rounds`
- `average_report_length`

---

## 八、运行产物与演示资产

### 运行工件
- 最新运行：`experiments/results/latest_run.json`
- 按主题保存的运行工件：`experiments/results/*_run.json`

### 面试演示文档
- 演示脚本：`docs/demo_script.md`
- 示例输出说明：`docs/example_output.md`

### 演示截图
- `docs/screenshots/dashboard_overview.png`
- `docs/screenshots/report_output.png`
- `docs/screenshots/evaluation_metrics.png`

如果需要重新生成截图素材：

```bash
python experiments/generate_demo_assets.py
```

---

## 九、面试时建议怎么讲

### 1. 为什么不用 LangChain 起步
因为这个项目的核心逻辑规模并不大，直接用 Python 实现更能展示：

- 工作流设计
- 工具调用协议
- 结构化输出
- Trace 与调试能力

### 2. 为什么不是完全自治多智能体
因为 research 类任务更偏线性依赖，workflow 更容易控制、调试和评测。

### 3. 为什么要做引用约束
因为没有引用约束的研究报告很容易被质疑“只是模型在编”，而引用可追溯是工程可信度的核心。

### 4. 为什么要做 Trace 和工件保存
因为 Agent 系统真正难的不是“写出来”，而是“出问题能不能定位、能不能复现、能不能解释”。

### 5. 为什么要做评测
因为没有评测，这个项目很容易被理解成“包装了一层工作流的 API Demo”。

---

## 十、这个项目最适合你的展示方式

如果你拿它去面试，最推荐的展示顺序是：

1. **先讲定位**：这是可验证的 research agent，不是简单问答机器人
2. **再讲链路**：Planner → Researcher → Writer → Reviewer
3. **再讲证据链**：引用、reference、latest_run.json
4. **再讲评测**：retrieval eval、workflow ablation、metrics summary
5. **最后讲取舍**：为什么现在先做 workflow，而不是完全自治多智能体

---

## 十一、补充说明

当前项目支持两种运行模式：

1. **轻量默认模式**：不依赖重型本地模型，直接可跑，适合演示
2. **增强模式**：安装 `requirements-rag.txt` 后，可启用更真实的 embedding / reranker 路径

这保证了：

- 默认情况下项目能快速启动
- 需要更强检索质量时，也有升级路径

---

## 十二、目标

这个项目最终不是为了做成“大而全平台”，而是为了做成一个：

- 能运行
- 能演示
- 能解释
- 能量化
- 能经得住面试官追问

如果你把这个项目讲清楚，它已经足够支撑大模型开发和 Agent 应用岗位的面试展示。
