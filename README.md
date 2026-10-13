<h1 align="center">智能客服工作台</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License Apache 2.0"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-%E2%89%A51.0-purple.svg" alt="LangGraph"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-%E2%89%A50.115-009688.svg" alt="FastAPI"></a>
</p>

<p align="center"><em>意图分类 · 专家路由 · 受控转接 · RAG 检索增强 · 工具调用 · 会话持久化</em></p>

## 项目概述

基于 LangGraph 的多智能体客服系统。用户查询经意图分类后路由到五个领域专家（产品 / 技术支持 / 账单 / 投诉 / 综合），专家可检索本地知识库增强回答、调用业务工具，并按白名单规则单跳转接其他专家；对话状态由 SQLite checkpointer 持久化，跨请求可续聊。整个服务以 FastAPI 单进程运行，前端内置于同一服务，零额外部署组件。

## 运行效果

以下为系统实测截图，覆盖核心链路（本地知识库检索、业务工具调用、安全护栏、多轮上下文、转接路由）：

### 工作流全图
从编译后的真实状态图导出：实线为意图路由，虚线为专家间白名单转接路径。
![工作流](./doc/workflow.png)

### 知识库检索增强回答
产品问答由本地知识库驱动，参数与政策均来自检索结果。
![知识库检索增强回答](./doc/demo-general-rag.png)

### 技术支持：查订单 + 建工单
按客户诉求连续调用业务工具，生成工单编号并给出处理时效。
![工具调用](./doc/demo-tech-tool-calling.png)

### 越界拒答护栏
非业务诉求在分类节点直接短路，不进入业务智能体。
![越界拒答](./doc/demo-out-of-scope-guard.png)

### 多轮对话上下文
基于 checkpointer 的跨请求会话记忆，支持指代与追问。
![多轮对话](./doc/demo-multi-turn-context.png)

### 投诉处理（混合诉求路由）
投诉与退款查询混合诉求正确路由至投诉处理专家。
![投诉处理](./doc/demo-complaint-handoff.png)

## 核心设计

### 意图分类：三层防御
1. **结构化输出主路径**：`with_structured_output` 直接产出标签，脏输出概率最低；
2. **文本模式降级**：结构化解析失败时退回纯文本分类 + 解析；
3. **标签规范化兜底**：`normalize_classifier_label` 对未知标签做别名归一，仍失败则落入 `general_inquiry`，任何情况下路由不会悬空。

### 检索增强与工具调用
- **知识库检索**：`retrieval/` 自实现的 FAISS 封装（`IndexFlatIP` 余弦检索 + JSON 元数据），支持两种嵌入来源：OpenAI 兼容端点（`EMBEDDING_PROVIDER=api`）或本地 sentence-transformers 模型（`EMBEDDING_PROVIDER=local`，可指向本地模型目录）；索引未就绪时返回空串，Agent 自动降级为关键词匹配。
- **业务工具**：mock 订单查询、工单创建工具，经 `BaseAgent` 的有界工具循环执行（上限 3 轮），工具报错以文本回传给 LLM 自行解释，而非直接抛给用户。

### 受控转接（handoff）
专家间的协作不做全动态 supervisor——每轮多一次 LLM 调用在本场景是负收益。改为**白名单单跳转接**：目标必须在该专家的 `handoff_targets` 白名单内，`handoff_count` 上限 1 杜绝 A→B→A 循环，转接轮不写对话记录，最终回答由目标专家产出。

### 会话持久化
对话轮次写入 LangGraph checkpointer 状态（SQLite），以 `session_id` 为 thread_id 跨请求续聊；轻量会话注册表（`session_store.py`）负责会话列表、消息计数等元信息，两者职责分离。

## 项目结构

```
customer-service-ai-agent/
├── app.py                        # FastAPI 入口：图进程内运行 + SSE 流式接口
├── multi_agent_customer_service.py  # LangGraph 主图：分类 → 路由 → 转接 → 汇总
├── llm.py                        # ChatOpenAI 工厂（流式 / function calling / 无 Key 降级）
├── config.py                     # 配置（全部可由 .env 覆盖）
├── session_store.py              # 会话注册表（列表 / 计数 / 元信息）
├── multi_agents/                 # 五个领域专家，统一继承 BaseAgent
│   ├── base_agent.py             # 共享能力：上下文组装、检索、有界工具循环、转接
│   ├── product_agent.py / tech_agent.py / billing_agent.py
│   ├── complaint_agent.py / general_agent.py
├── tools/
│   ├── query_tools.py            # 意图分类工具与标签规范化
│   └── business.py               # mock 业务工具（查订单 / 建工单）
├── retrieval/
│   ├── vector_store.py           # FAISS 封装：构建与检索（api / local 双嵌入通道）
│   ├── build.py                  # 索引构建 CLI
│   └── kb_docs/                  # 知识库种子文档（产品 / 售后政策）
├── evaluation/
│   ├── golden_set.jsonl          # 30 条评测用例（6 类）
│   └── run_eval.py               # 评测脚本：分类正确率 / 拒答召回 / 延迟 P95
├── tests/                        # 31 条离线 pytest（不依赖 API Key）
├── templates/index.html          # 内置 Web 前端
├── Dockerfile / docker-compose.yml
└── env_example.txt               # 环境变量模板
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp env_example.txt .env   # Windows: copy env_example.txt .env
```
编辑 `.env` 填入 `OPENAI_API_KEY`（默认指向硅基流动，任何 OpenAI 兼容端点均可）。

### 3. 构建知识库索引（可选）
```bash
python -m retrieval.build
```
未构建索引时系统照常运行，回答退化为关键词匹配。使用本地嵌入时，先安装 `sentence-transformers` 并在 `.env` 设置 `EMBEDDING_PROVIDER=local` 与 `EMBEDDING_LOCAL_MODEL`。

### 4. 启动服务
```bash
python app.py
```
浏览器访问 `http://localhost:5000`。未配置 Key 时服务照常启动，回答走降级分支，便于离线调试界面与路由。

### Docker 一键启动
```bash
docker compose up --build
```
运行时产物（对话检查点、知识库索引、评测报告）通过 `./data` 卷落盘宿主机。容器内建索引：
```bash
docker compose exec app python -m retrieval.build
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | Web 前端页面 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/test` | 环境自检（图编译 / Key 配置 / checkpointer） |
| POST | `/api/chat` | 同步聊天 |
| POST | `/api/chat/stream` | SSE 流式聊天 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{id}` | 会话详情（含完整对话历史） |
| DELETE | `/api/sessions/{id}` | 删除会话（含 checkpointer 状态清理） |

## 评测

评测体系覆盖四个维度，指标全部来自真实测量：

| 维度 | 指标 | 运行方式 | 依赖 |
|---|---|---|---|
| 意图路由 | 正确率、越界拒答召回、零误拒 | `python -m evaluation.run_eval` | API Key |
| 检索质量 | Recall@1、Recall@3、MRR、检索延迟 P95 | `python -m evaluation.run_metrics_eval retrieval` | 已建索引 |
| 工具调用 | 工具选择准确率、参数抽取准确率 | `python -m evaluation.run_metrics_eval tool` | API Key |
| 生成忠实度 | 忠实度得分、幻觉率（未支持声明占比） | `python -m evaluation.run_metrics_eval faithfulness` | API Key + 索引 |
| 任务完成率 | 端到端达成用户诉求的比例 | `python -m evaluation.run_task_eval` | API Key + 索引 |
| 异常恢复率 | 7 类故障注入后按设计降级/自愈的比例 | `python -m evaluation.run_resilience_eval` | 离线 |
| 任务成本 | 单次任务 token 消耗与平均成本 | `python -m evaluation.run_cost_eval` | API Key |

- 检索评测基于 15 条 golden 查询（6 份知识库文档全覆盖），统计期望文档在 Top-K 的命中与排名，检索延迟与生成延迟分开计量。
- 工具评测基于 8 条用例（含不存在订单的错误处理路径与"不该调工具"的反例），以记录器包装业务工具捕获真实调用参数，同时验证工具选择与参数抽取。
- 生成忠实度采用 LLM-as-judge：把回答拆分为事实性声明，逐条对照检索到的参考材料核验；忠实度 = 被支持声明占比，幻觉率 = 1 − 忠实度（与 RAGAS Faithfulness 同思路的轻量实现）。
- 指标汇总输出到 `evaluation/metrics_report.md`，失败用例（未命中查询、误调用、疑似幻觉声明）逐条列明；指标计算逻辑为纯函数、离线可测，非零退出码可接 CI。

### 实测结果

环境：LLM 为硅基流动 `deepseek-v4-flash`，嵌入为本地 `bge-large-zh-v1.5`（CPU），端到端跑完整图。

| 指标 | 结果 |
|---|---|
| 意图路由正确率 | 28/30 = **93.3%** |
| 越界拒答正确率 | **100%**（5/5），正常请求零误拒（25/25） |
| 检索质量 | Recall@3 **100%** / Recall@1 93.3% / MRR 0.956，检索延迟 P95 185ms |
| 工具调用 | 选择准确率 **87.5%**、参数抽取准确率 **100%** |
| 任务完成率 | **83.3%**（12 条端到端用例） |
| 回答忠实度 | **84.1%**（幻觉率 15.9%，LLM-as-judge 保守下界） |
| 异常恢复率 | **100%**（7/7 故障注入场景） |
| 单次任务平均成本 | ¥0.0107（含全部 LLM 调用环节） |
| 测试覆盖率 | 60%（全仓语句；核心编排/工具模块 90-100%） |
| 端到端延迟 | 平均 8.7s / P95 15.0s |

分类别：product 8/8、tech 5/5、complaint 4/4、general 3/3、out_of_scope 5/5 全对，billing 3/5——两条误判（订单进度查询、退货条件咨询）落入 general_inquiry，属分类边界案例，可通过强化分类 prompt 的账单类描述优化。延迟主要来自本地嵌入模型 CPU 推理与 LLM 端点往返，可切换 GPU 或更小嵌入模型改善。

## 测试

```bash
python -m pytest tests/ -q
```

52 条测试全部离线可跑（不依赖 API Key）：分类三层防御、图编译与护栏短路、转接路由与循环防护、工具调用循环（脚本化假 LLM 驱动）、检索降级、嵌入工厂双通道、评测指标纯函数与 golden 文件校验、日志配置行为、七类故障注入恢复场景。

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | 硅基流动 + Qwen3-8B | LLM 端点 |
| `LLM_ENABLE_THINKING` | `false` | Qwen3 系列思考模式开关 |
| `EMBEDDING_PROVIDER` | `api` | `api`（OpenAI 兼容端点）或 `local`（sentence-transformers） |
| `EMBEDDING_MODEL` / `EMBEDDING_LOCAL_MODEL` / `EMBEDDING_DEVICE` | bge-large-zh-v1.5 | 嵌入模型与设备 |
| `RETRIEVAL_TOP_K` | `3` | 检索返回片段数 |
| `CHECKPOINT_DB_PATH` | `./data/checkpoints.sqlite` | 对话检查点 |
| `APP_HOST` / `APP_PORT` | `0.0.0.0:5000` | 服务监听 |
| `LOG_LEVEL` | `INFO` | 日志级别（诊断信息分级：warning=已自动降级兜底，error=请求内失败已返回兜底回复） |

完整列表见 `env_example.txt`。

## 技术栈

LangGraph（工作流编排 + checkpointer）· LangChain Core（消息与结构化输出）· FAISS（本地向量检索）· FastAPI（单服务 + SSE）· pytest（离线回归）

## License

Apache 2.0
