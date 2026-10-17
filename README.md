# fire-safety-rag · 工厂消防知识问答助手

面向工厂安全员与一线员工的**消防知识问答助手**。导入消防法规、应急预案、培训资料与扫描件 PDF，
用 RAG 给出**有出处、可追溯、不编造**的回答。

- 目标阶段：**第一批**（基础 RAG + 混合检索 + Rerank + 评估基线）+ **第二批**（扫描件 OCR、HyPE、Query Transformation、语义缓存、RAGAS）
- 架构：前端 Vercel（Next.js） + 后端 Render（FastAPI 容器） + Neon PostgreSQL(pgvector) + DeepSeek API

> 📖 **要读代码，先看 [`docs/项目导览.md`](docs/项目导览.md)** —— 目录结构、按目的的阅读路径、
> 核心文件逐个说明、`.env` 开关一览、脚本速查、已知限制清单。
> 评估策略与各阶段结论见 [`docs/评估方案.md`](docs/评估方案.md)。

---

## 技术栈总览

| 层 | 选型 | 说明 / 为什么这么选 |
|---|---|---|
| 语言运行时 | Python **3.12** + Node **24** | 3.13 上 PaddleOCR 支持不完整；Node 只用于前端构建 |
| Web 框架 | **FastAPI** + Uvicorn | 处理器写成**同步 `def`**（整条链路是阻塞式的，FastAPI 会丢线程池；写成 `async def` 会踩 nested async，见踩坑 #14） |
| RAG 框架 | **LlamaIndex 0.12** | 数据层（NodeParser/VectorStoreIndex）+ 检索层（QueryFusionRetriever/Reranker）都复用它 |
| LLM | **DeepSeek `deepseek-chat`** | OpenAI 兼容协议，用 `OpenAILike` 接入；HyPE / 指代消解 / 生成 / RAGAS 裁判共用 |
| Embedding | **BAAI/bge-m3**（1024 维） | 本地 HuggingFace 加载（CUDA，batch 64，约 660 条/秒）或切 SiliconFlow API；多语言、长文本 |
| Rerank | **BAAI/bge-reranker-v2-m3** | 交叉编码器精排，本地 CUDA + fp16；投入产出比最高的一个环节 |
| 向量库 | **PostgreSQL + pgvector**（Neon） | LlamaIndex `PGVectorStore`；HNSW(cosine) m=16 / ef_construction=64 / ef_search=40 |
| 稀疏检索 | **BM25 (bm25s)** | 消防术语精确匹配（`GB 50016`、`甲类厂房`、`4.0m`）；中文用 **2-gram 字符分词**兜底（无需 jieba 词典） |
| 融合策略 | **RRF**（倒数排名融合） | 向量相似度 0.83 与 BM25 得分 12.7 量纲不可比，RRF 天然规避 |
| ORM / 迁移 | **SQLAlchemy 2.0** + Alembic | 业务表（documents/chunks/sessions/messages/eval_records） |
| 配置 | **pydantic-settings** | 全部可调参数集中在 `app/config.py`，换模型只改 `.env` |
| 文档解析 | **pdfplumber / PyMuPDF / PaddleOCR / python-docx / BeautifulSoup4 / MinerU** | 三级解析阶梯 + 离线 MinerU 预处理（PDF→Markdown） |
| 缓存 | **内存 LRU**（可选 Redis） | 语义缓存：问题向量化后比余弦，阈值 0.95；Redis 不可用自动降级 |
| 会话记忆 | **PostgreSQL** | 可审计；批量写入（`add_turn`）压跨国往返 |
| 评估 | **自建 MRR / HitRate / Recall / nDCG** + **RAGAS** | 检索层与生成层分开评估 |
| 可观测 | LangFuse（待接入） | 第二批 |
| 前端 | **Next.js 14** + React 18 | 纯 CSS 设计系统（无 UI 框架依赖），4 个 Tab |
| 部署 | 前端 **Vercel** / 后端 **Render**（Docker） | 免费套餐跑不动本地模型，线上走 SiliconFlow + 关闭 Rerank |

---

## 两条核心链路

### 链路一：数据导入（离线，一次性或增量）

```
素材来源
  ├─ 上传接口   POST /api/ingest/upload        routes_ingest.py（multipart）
  ├─ 批量 CLI   scripts/ingest_cli.py          --dir / --file [--force|--reset]
  ├─ 网页/文本  POST /api/ingest/url | /text
  └─ 复杂 PDF   scripts/mineru_convert.py      MinerU 把 PDF 转 Markdown 再入库
        │
        ▼
┌─ [1] 落盘 + 幂等 ────────────────────────────────────────────────────────────┐
│  data/uploads/ + 文件 SHA-256(file_hash)                                     │
│  同一文件重复上传直接跳过（除非 --force / force=true）                        │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [2] 解析：三级阶梯 ─────────────────────────────────────────────────────────┐
│  loader.load_file() → 按后缀分发                                             │
│                                                                              │
│   PDF ─► pdfplumber 抽文字层 ──► quality.text_quality_score() 质量分          │
│            │                        │                                        │
│            │                  合格 ─┴─► 直接使用（quality ≈ 0.9~1.0）         │
│            │                        │                                        │
│            │                  乱码 ────► PyMuPDF 重抽，取质量更高者           │
│            │                        │                                        │
│            └──────────────► 仍不合格 或 页均字符 < 50                         │
│                                     └─► PaddleOCR 逐页 OCR（按页磁盘缓存）     │
│   DOCX ─► python-docx（Heading 样式转 ##）                                    │
│   HTML ─► BeautifulSoup4    TXT/MD ─► 直读    图片 ─► PaddleOCR               │
│                                                                              │
│   ⚠️ 为什么需要"质量分"：方正书版国标 PDF 有文字层但抽出来是                   │
│      FGM(cid:212)T 这种乱码，只按字符数判断会把垃圾静默入库                    │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [3] 切块 ───────────────────────────────────────────────────────────────────┐
│  chunker.chunk_document()                                                    │
│  ① split_by_heading()：识别 第X章 / 第X条 / 3.2.1 / （三）→ 标题链            │
│  ② 标题链拼进正文 —— 让 embedding 也"看到"章节语义                            │
│  ③ SemanticSplitterNodeParser(buffer=1, 断点百分位=95%) 语义断点              │
│  ④ 兜底 SentenceSplitter(512/64)；丢弃 < 20 字的碎片                          │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [4] HyPE（第二批）─────────────────────────────────────────────────────────┐
│  hyphe.attach_hype()：DeepSeek 为每块生成 3 个**口语化假想问题**              │
│  → chunks.hypothetical_questions + 节点 metadata                              │
│  → 这些问题会参与 embedding（提升口语提问的召回），但不污染正文与答案          │
│  例：正文"每月进行一次外观检查" → "灭火器多久检查一次啊？"                     │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [5] 落业务表（SQLAlchemy → Neon PostgreSQL）────────────────────────────────┐
│  documents：状态机 pending → parsing → chunking → embedding → indexed/failed │
│  chunks   ：content / heading_path / page_no / token_count / 假想问题         │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [6] 向量化入库 ────────────────────────────────────────────────────────────┐
│  indexer._build_nodes() → TextNode                                           │
│  ・NodeRelationship.SOURCE = document_id（否则向量表 document_id 是 'None'）  │
│  ・bge-m3（device=cuda, batch=64）→ PGVectorStore                            │
│  ・表 data_fire_chunks(node_id, text, metadata_, embedding vector(1024))     │
│    自动建 HNSW 索引（vector_cosine_ops）                                      │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [7] 缓存失效 ──────────────────────────────────────────────────────────────┐
│  invalidate_index_cache() + invalidate_retriever_cache() + clear_cache()     │
│  知识库变了，索引单例 / 检索器缓存 / 语义缓存都必须失效，否则查不到新数据      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 链路二：查询问答（每次提问，稳态约 12 秒）

```
用户提问
  POST /api/query                    routes_query.py（同步 def → FastAPI 线程池）
        │
        ▼
┌─ [0] 会话 ───────────────────────────────────────────────────────────────────┐
│  SessionStore.get_or_create(session_id, user_id)                             │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [1] 语义缓存 ──────────────────────────────────────────────────────────────┐
│  cache.get_cached()：单例 bge-m3 向量化问题 → 与缓存条目比余弦               │
│  命中（≥ 0.95）→ 直接返回 cached=true（"灭火器多久检查一次" ≈ "检查周期是？"）│
└──────────────────────────────────────────────────────────────────────────────┘
        │ 未命中
        ▼
┌─ [2] 指代消解 ──────────────────────────────────────────────────────────────┐
│  _condense_question()：有多轮历史时，先用 DeepSeek 把"那它呢？"补全成        │
│  可独立检索的问题（否则检索必然答偏）                                        │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [3] 查询改写（可选，默认关）───────────────────────────────────────────────┐
│  query_transform.rewrite_queries()：Step-back 上位问题 / HyDE 假想答案        │
│  ⚠️ 代价 = 一次 LLM + 一次完整检索，跨国库上性价比低，默认关闭                │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [4] 混合检索 + 精排 ───────────────────────────────────────────────────────┐
│  get_hybrid_retriever()  ← 进程内缓存单例（每请求重建要多花约 8 秒）          │
│                                                                              │
│   稠密：PGVectorStore HNSW 余弦                         ┐                    │
│         （可带 doc_category 元数据过滤，过滤在库端）    ├─► RRF 融合          │
│   稀疏：BM25(bm25s) + 中文 2-gram 分词                  ┘   QueryFusionRetriever│
│                                                              mode=rrf, top10  │
│                          ▼                                                   │
│   精排：bge-reranker-v2-m3 交叉编码器（CUDA + fp16）→ top 5                  │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [5] 组装提示词 ────────────────────────────────────────────────────────────┐
│  prompt.build_qa_messages()：三条硬规则                                     │
│   ① 只用【参考资料】作答   ② 资料不足就明说"无法确定"   ③ 必须标注 [编号]     │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [6] 生成 ──────────────────────────────────────────────────────────────────┐
│  llm.chat_direct()：DeepSeek deepseek-chat（同步 httpx）                     │
│  也可用 stream_chat() 做 SSE 流式输出                                        │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
┌─ [7] 落库 + 写缓存 ─────────────────────────────────────────────────────────┐
│  SessionStore.add_turn()：一次提交写"一问一答"两条消息（省跨国往返）          │
│  cache.set_cached()：缓存问题向量与答案                                      │
└──────────────────────────────────────────────────────────────────────────────┘
        ▼
响应 data{ answer, sources[](条款/页码/相关度/原文), cached, latency_ms, transformed_queries }
```

### 链路三：评估（离线）

```
检索评估  POST /api/eval/retrieval
  golden 问答集 → retriever.retrieve() → 与 ground_truth_ids 比对
  → HitRate@K / MRR@K / Recall@K / nDCG@K → eval_records（含配置快照，可跨版本对比）

生成评估  POST /api/eval/generation
  自动跑一遍问答链路补齐 answer/contexts → RAGAS
  → context_precision / context_recall / faithfulness（faithfulness 是消防场景红线）

评估集生成  POST /api/eval/build-golden-set
  从库中抽样切块 → DeepSeek 出题 → 草稿 JSONL（**必须人工校对**）
```

### 实测耗时分解（RTX 4060 Ti + Neon 新加坡）

| 环节 | 耗时 | 说明 |
|---|---|---|
| 问题向量化 | **0.19s** | GPU，可忽略 |
| 精排 20→5 条 | **1.4s** | GPU + fp16 |
| DeepSeek 生成 | **2.3s** | 网络 |
| 混合检索（含十几次 DB 往返） | **2.4~5.4s** | 瓶颈：跨境 SQL |
| 会话读写（4 条 SQL） | **2.6s** | 瓶颈：跨境 SQL |
| 单条 `SELECT 1` | **0.86s** | 跨境 RTT 0.4s |
| **稳态合计** | **≈12.5s** | 首次另加约 50s 模型加载 |
| 语义缓存命中 | **≈3.8s** | 只剩会话写库 |

> 结论：**模型不是瓶颈（GPU 上合计约 4 秒），瓶颈是跨境数据库往返**。
> 优化路径见「性能」一节（本地 Postgres 可到 ~4 秒）。

---

## 一、环境准备（分平台）

### 1. 创建虚拟环境

```bash
# 在项目根目录 fire-safety-rag/ 下执行
python -m venv .venv
```

> 作用：把本项目的依赖装进独立目录 `.venv/`，与系统 Python 隔离，避免版本冲突。
> 要求 Python **3.11 或 3.12**（3.13 上 PaddleOCR/PaddlePaddle 支持不完整）。

### 2. 激活虚拟环境

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Windows Git Bash
source .venv/Scripts/activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

> 激活后命令行前面会出现 `(.venv)`。若 PowerShell 报"禁止运行脚本"，先执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. 升级 pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

> 作用：新版 pip 的依赖解析器更快更准，能避免很多"装到一半冲突"的问题。

### 4. 安装依赖

```bash
# 先装 CPU 版 torch（体积小、Render 免费套餐才跑得动）
pip install torch --index-url https://download.pytorch.org/whl/cpu

cd backend
pip install -r requirements.txt
```

需要 RAGAS 的完整能力时再补：

```bash
pip install langchain-openai langchain-community datasets
```

### 5. 配置环境变量

```bash
cd backend
cp .env.example .env       # Windows: copy .env.example .env
# 然后编辑 .env，至少填 DEEPSEEK_API_KEY 与 DATABASE_URL
```

### 6. 初始化数据库

```bash
# 方式 A：直接跑 SQL（Neon 控制台 SQL Editor 粘贴 scripts/init_db.sql）
psql "$DATABASE_URL" -f scripts/init_db.sql

# 方式 B：Alembic 迁移（推荐，生产用这个）
alembic revision --autogenerate -m "init schema"
alembic upgrade head
```

---

## 二、运行后端

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 接口文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

批量导入资料（可选）：

```bash
python scripts/ingest_cli.py --dir ../资料/法规 --category law
```

运行测试：

```bash
pytest -q                      # 单元测试，无需网络与模型
python scripts/test_api.py     # API 冒烟测试，需要 Neon + DeepSeek
```

### 测试分层（从便宜到贵，建议按顺序跑）

| 层级 | 命令 | 覆盖内容 | 依赖 |
|---|---|---|---|
| 1. 单元测试 | `pytest -q` | 切块、乱码判定、评估指标、缓存、健康检查（18 项） | 无 |
| 2. 环境自检 | `python scripts/selfcheck.py` | 连接串分支、关键配置（脱敏）、实连 Neon | 数据库 |
| 3. 设备/缓存 | `python scripts/bench_embed.py` | torch 是否 CUDA 版、device 是否 cuda、吞吐 | 本地模型 |
| 4. 数据库体检 | `python scripts/db_debug.py` | 向量行数与切块数是否一致、HyPE 是否落库 | 数据库 |
| 5. **API 冒烟** | `python scripts/test_api.py "你的问题"` | **11 项接口行为**：健康检查、资料列表、首次问答、引用、语义缓存命中、多轮指代消解、会话落库、鉴权 401/200、清缓存 | 数据库 + LLM |
| 6. 人工验收 | 浏览器打开 `/docs` → `POST /api/query` | 真实提问体验、审读答案质量 | 同上 |

> `test_api.py` 用 TestClient 在**进程内**跑完整的 ASGI 路由，所以不需要先启动 uvicorn；
> 也正因为它走的是真实 ASGI 路径，才能抓到"同步处理器写成 async def"这类
> 命令行脚本永远重现不了的 bug（见踩坑记录 #14、#15）。

---

## 三、目录结构与文件职责

```
fire-safety-rag/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI 入口：CORS、路由挂载、健康检查、生命周期
│   │   ├── config.py                    # pydantic-settings 全局配置，所有可调参数集中处
│   │   ├── dependencies.py              # LLM/Embedding/Reranker 懒加载单例注入
│   │   ├── api/
│   │   │   ├── routes_ingest.py         # 上传/删除/重建索引/资料列表
│   │   │   ├── routes_query.py          # 问答接口（含会话记忆与语义缓存）
│   │   │   ├── routes_eval.py           # 触发检索评估与 RAGAS 评估
│   │   │   └── routes_admin.py          # 统计、配置查看、缓存清理
│   │   ├── core/
│   │   │   ├── ingestion/
│   │   │   │   ├── loader.py            # PDF/Word/HTML/TXT 解析，PDF 三级阶梯降级
│   │   │   │   ├── quality.py           # ★新增：文本质量分，识别"有文字层但是乱码"
│   │   │   │   ├── ocr.py               # PaddleOCR 扫描件识别 + 按页缓存 + 2.x/3.x 适配
│   │   │   │   ├── chunker.py           # 语义切块 + 标题链拼接
│   │   │   │   ├── hyphe.py             # HyPE：为每块生成 3 个假想问题
│   │   │   │   ├── embedder.py          # bge-m3 本地 / SiliconFlow API 两条路
│   │   │   │   └── indexer.py           # ★新增：解析→切块→HyPE→入库 的编排流水线
│   │   │   ├── retrieval/
│   │   │   │   ├── hybrid.py            # BM25 + 向量混合检索组装（含中文分词器）
│   │   │   │   ├── fusion.py            # QueryFusionRetriever / RRF 融合、结果转 source
│   │   │   │   ├── reranker.py          # bge-reranker-v2-m3 精排与包装
│   │   │   │   └── query_transform.py   # HyDE + Step-back 查询改写
│   │   │   ├── generation/
│   │   │   │   ├── prompt.py            # 消防问答 Prompt（含"不许编造"约束）
│   │   │   │   └── llm.py               # DeepSeek 封装（OpenAILike + 裸 HTTP + 流式）
│   │   │   ├── evaluation/
│   │   │   │   ├── retriever_eval.py    # MRR / HitRate / Recall / nDCG 自建评估
│   │   │   │   └── ragas_eval.py        # RAGAS: context_precision/recall/faithfulness
│   │   │   └── memory/
│   │   │       └── session_store.py     # 会话与消息读写、指代消解所需历史、滚动摘要
│   │   ├── models/
│   │   │   ├── document.py              # 文档/切块的领域模型与解析中间产物
│   │   │   ├── request.py               # 全部 POST 请求体
│   │   │   └── response.py              # 统一响应包装与各接口响应体
│   │   ├── db/
│   │   │   ├── session.py               # SQLAlchemy 引擎/Session/建表/ping
│   │   │   ├── models.py                # documents/chunks/chat_sessions/chat_messages/eval_records
│   │   │   ├── vector_store.py          # ★新增：LlamaIndex PGVectorStore 构建与删除
│   │   │   └── migrations/              # Alembic：env.py、script.py.mako、versions/
│   │   └── utils/
│   │       ├── cache.py                 # 语义缓存（Redis + 余弦相似度，降级内存 LRU）
│   │       └── logging.py               # 日志格式、噪音压制、计时器
│   ├── tests/                           # 切块/评估指标/缓存/健康检查单测
│   ├── data/
│   │   ├── uploads/                     # 上传原文件
│   │   ├── indexes/                     # OCR 缓存等中间产物
│   │   └── eval/                        # 评估集（JSONL）
│   ├── scripts/
│   │   ├── init_db.sql                  # 扩展 + 兜底建表 SQL
│   │   ├── ingest_cli.py                # 批量导入命令行
│   │   ├── mineru_convert.py            # ★新增：用 MinerU 把复杂 PDF 转 Markdown（可选）
│   │   ├── selfcheck.py                 # 环境自检：连接串分支/关键配置/连库验证
│   │   ├── db_debug.py                  # ★新增：数据库体检（向量/业务表是否一致、HyPE 是否落库）
│   │   ├── bench_embed.py               # ★新增：device 检查与向量化吞吐测试
│   │   ├── test_vector_store.py         # ★新增：PGVectorStore 同步/异步连接回归
│   │   ├── test_query.py                # ★新增：端到端问答冒烟（检索→生成→会话记忆）
│   │   ├── test_api.py                  # ★新增：进程内 API 冒烟测试（11 项，一条命令）
│   │   ├── probe_pdf.py                 # 探测 PDF 页数与文字层质量
│   │   ├── probe_parsers.py             # 对比 pdfplumber / PyMuPDF 抽取效果
│   │   ├── probe_quality.py             # 乱码检测定标（含正常文本对照）
│   │   └── test_ocr.py                  # 单文件 OCR 冒烟测试
│   ├── alembic.ini
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/                            # Next.js 单页前端（本地即用，可选部署到 Vercel）
│   ├── app/
│   │   ├── page.js                      # 外壳：顶栏 + Tab 导航 + Toast + 会话/设置状态
│   │   ├── layout.js                    # 根布局（引入全局样式）
│   │   ├── globals.css                  # 设计系统（CSS 变量 + 组件样式）
│   │   ├── lib/api.js                   # API 封装（统一解包/报错、带进度的上传、localStorage）
│   │   └── components/
│   │       ├── ChatTab.js               # 问答：对话流、出处卡片、载入计时
│   │       ├── SessionsTab.js           # 会话列表：继续对话、删除
│   │       ├── DocsTab.js               # 资料库：拖拽导入、进度、状态、切块预览、删除、重建
│   │       └── SettingsTab.js           # 设置：检索参数、服务配置、统计、清缓存
├── render.yaml                          # Render Blueprint 一键部署
└── README.md
```

> ★新增文件说明：原目录树里 `ingestion/` 只有各步工具函数，缺少把它们串起来的编排者，
> 因此新增 `indexer.py`；同时把"向量库构建"从检索模块里独立成 `db/vector_store.py`，
> 让 ingestion 与 retrieval 都能复用同一份 PGVectorStore 配置。

---

## 四、核心链路说明

### 1. 导入链路

`上传 → loader 解析 → (扫描件则 PaddleOCR) → chunker 语义切块 + 标题拼接 → HyPE 生成假想问题 → bge-m3 向量化 → PGVectorStore 落库 + 业务表落库`

关键点：

- **标题链拼接**：每个 chunk 的正文前加上 `第三章 消防设施 > 3.2 灭火器配置`，让向量也"看到"章节语义；否则"不得少于 2 具"这种片段脱离上下文后毫无检索价值。
- **HyPE**：正文之外，把"这段文字能回答的 3 个问题"也拼进被索引的文本（`build_hype_text`），检索时更容易匹配用户口语化提问。**展示与生成仍用原文**，不污染答案。
- **幂等**：以文件 SHA-256 去重，重复上传直接跳过。

### 2. 检索链路

`查询 → (多轮时指代消解) → (可选 Step-back/HyDE 改写) → 向量 top10 + BM25 top10 → RRF 融合 → Reranker 精排 top5 → 拼 Prompt → DeepSeek 生成 → 标注 [1][2] 出处`

- **为什么用 BM25**：消防资料里大量精确串（`GB 50016`、`甲类厂房`、`4.0m`），纯向量检索对这类词召回差，稀疏检索正好互补。
- **中文分词**：没装 jieba 时自动用字符 2-gram，保证开箱可用。
- **RRF 融合**：向量相似度 0.83 与 BM25 得分 12.7 不可比，倒数排名融合天然规避量纲问题。
- **精排**：交叉编码器只跑召回后的 20 条，精度提升明显、延迟可控。

### 3. 生成链路

`prompt.py` 的三条硬规则：只用参考资料作答、资料不足就明说、结论必须标注 `[编号]`。
消防是安全相关场景，**宁可回答"资料中没有"，也不能编造数值**。

### 4. 会话记忆

会话与消息存 PostgreSQL（可审计）。多轮提问先用 `condense` 把"那它要几个？"补全成独立问题再检索，
否则检索必然答偏。长会话超过阈值时把早期对话压缩成摘要存 `chat_sessions.summary`。

### 5. 复杂 PDF 怎么办（扫描件 / 国标乱码 / 表格多）

入库前先探一下这份 PDF 属于哪一类：

```powershell
python scripts/probe_pdf.py "D:\资料\某国标.pdf"     # 页数、文字层字符数、是否扫描件
```

**解析三级阶梯**（`loader.py` 自动执行，无需人工干预）：

| 级别 | 触发条件 | 手段 | 说明 |
|---|---|---|---|
| ① | 默认 | pdfplumber 抽文字层 | 常规 PDF 效果最好，还带表格能力 |
| ② | ①的质量分 < `MIN_TEXT_QUALITY` | 换 PyMuPDF 重抽 | 部分 CID 字体它能救回数字与标点 |
| ③ | ②仍不合格，或页均字符 < 50 | PaddleOCR 逐页 OCR | 扫描件、方正书版乱码 PDF |

**为什么要做"质量分"判定**：国标 PDF 常用方正书版（`FzBookMaker` 字体 + `Identity-H` 编码、
无 ToUnicode 映射），文字层"存在"且字符数很多，但抽出来是：

```
pdfplumber : 犌犅／犜３８３１５—２０１９ \n FGM(cid:212)T(cid:213)5}(cid:204)!FG(M}`O
PyMuPDF    : ａ）　FGMÔTÕ5}Ì!FG(M}ÁO，"a®Ü8G&，tÚ３００ｍ～５００ｍ01(UY
```

只按"字符数够多 = 有文字层"判断，这种文件会被当正常文档入库，**结果是一库无法检索的
垃圾而且不报错**。判定细节与实测标定见 `app/core/ingestion/quality.py` 顶部注释。

#### 方案 A：PaddleOCR（已内置，进程内自动跑）

无需额外操作，导入时自动走 OCR。注意几点：

```powershell
# 首次会下载检测/识别模型（约 50MB）；模型缓存在 %USERPROFILE%\.paddlex
python scripts/test_ocr.py "D:\资料\某扫描件.pdf" 2    # 只跑前 2 页做冒烟
```

- PaddleOCR **2.x 与 3.x 的 API 完全不同**，本项目按主版本号自动适配；
  3.x 下 `use_angle_cls` / `show_log` 已被移除，别按老教程改代码。
- 文字行方向模型 `PP-LCNet_x1_0_textline_ori` 在国内常下载失败，默认已关闭
  （`OCR_TEXTLINE_ORIENTATION=false`）；只有倒置/竖排扫描件才需要打开。
- **坑**：模型目录若只建了目录、没下到文件，PaddleX 会"以为已存在"并跳过下载，
  表现为 `FileNotFoundError: ...\inference.yml`。删掉那个空目录（或整个
  `%USERPROFILE%\.paddlex`）再重跑即可。
- 若遇到 `RuntimeError: PDX has already been initialized`，设环境变量
  `PADDLE_PDX_EAGER_INIT=false` 绕过（见 paddlex/utils/flags.py）。

#### 方案 B：MinerU（推荐用于"表格多 / 版面复杂 / 乱码严重"的资料）

MinerU 不依赖 PDF 文字层，走版面识别 + OCR，输出**结构化 Markdown**
（标题层级、表格、阅读顺序都对），而本项目的 chunker 正好按 `#` 标题链切块，
所以衔接是免费的：**PDF --MinerU--> Markdown --本项目 loader--> 入库**。

架构上刻意把 MinerU 放在**离线独立环境**，不进 FastAPI 进程：

```
[独立 venv: mineru]  PDF --> .md
                               |
[项目 venv: backend]  scripts/mineru_convert.py --> ingest_cli / indexer --> pgvector
```

理由：MinerU base 包是 ONNX + llama.cpp，standard 档模型约 2GB、最低 8GB 内存，
塞进服务里会让 Render 免费套餐直接 OOM；且它与 llama-index 依赖容易冲突。

安装与使用：

```powershell
# 1) 独立环境安装 MinerU（约 1~3GB 模型，首次运行自动下载）
pip install uv
uv venv .mineru --python 3.12
uv pip install --python .mineru "mineru>=4.0,<5"

# 2) 转换（可先只转前 10 页试质量）
.mineru\Scripts\mineru-kit.exe parse "D:\资料\某国标.pdf" -o "D:\资料\某国标.md" --tier basic

# 3) 用本项目脚本转换 + 直接入库（需要 mineru 在 PATH 里）
python scripts/mineru_convert.py "D:\资料\某国标.pdf" --tier basic --ingest --category law
```

档位选择：`basic` ≈ 0.8GB 模型 / 最低 2GB 内存，多数国标够用；
`standard` ≈ 2GB 模型 / 最低 8GB 内存，版面复杂（多栏、表格密集）时更稳。

> 转换出的 `.md` 建议保留在版本库或资料目录里：可人工复核、可局部修订，
> 比"每次导入都跑一遍模型"更可控。

### 6. 用上显卡（本地 embedding / rerank）

默认 `EMBEDDING_DEVICE=auto`：检测到可用 CUDA 就走 GPU，否则回落到 CPU。
**但只有 CUDA 版 torch 才有 GPU 能力**——如果装的是 CPU 版（版本号带 `+cpu`），
`torch.cuda.is_available()` 恒为 `False`，auto 也只会用 CPU。自检：

```powershell
python scripts/bench_embed.py     # 打印 torch/CUDA 状态并跑吞吐测试
```

#### 换成 CUDA 版 torch

```powershell
# 1) 先确认环境里的 torch 是不是 CPU 版
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 输出形如 2.14.1+cpu None False  -> 需要换成 GPU 版

# 2) 探测哪个 CUDA 源有与你当前 torch 匹配的版本（cu130/cu126 等按需替换）
pip install --upgrade --dry-run --no-deps --index-url https://download.pytorch.org/whl/cu130 torch

# 3) 安装（必须带 --upgrade，否则 pip 会认为 torch 已满足而直接跳过）
pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch
```

⚠️ 两个必踩的坑：
- **不加 `--upgrade` 会“安装成功但什么也没变”**：pip 看到 `torch 2.x.x+cpu` 已满足就直接跳过，
  输出里只有一句 `Requirement already satisfied`。
- 只想用 CPU 时，别用 `pip install torch`（默认从 PyPI 拉的是 GPU 大包），
  用 `pip install torch --index-url https://download.pytorch.org/whl/cpu`。

#### 实测吞吐（RTX 4060 Ti 8GB，bge-m3）

| 设备 | 批大小 | 64 条文本向量化 | 吞吐 |
|---|---|---|---|
| CPU | 16 | 1.90s | 33.7 条/秒 |
| GPU | 64 | 0.10s | 653.6 条/秒 |

约 **19 倍**。`EMBEDDING_BATCH_SIZE` 在 GPU 上会自动提到 64（CPU 保持 16，批大反而更吃内存不加速）。

显存规划（8GB 足够）：bge-m3 约 2.3GB + bge-reranker-v2-m3 约 2.3GB，两者同时常驻也放得下。
`RERANK_USE_FP16` 在检测到 CUDA 时会自动生效（`FlagEmbeddingReranker` 没有 device 参数，它内部自行选卡）。

> 部署到 Render 免费套餐时请把 `EMBEDDING_DEVICE` 留空/`cpu` 或改用
> `EMBEDDING_PROVIDER=siliconflow`——云端实例没有你的本地显卡。

### 7. 语义缓存（第二批）

"灭火器多久检查一次" 与 "灭火器检查周期是多少" 字面不同、语义相同，普通 KV 缓存命不中。
语义缓存对问题向量化后比余弦相似度，`≥ 0.95` 直接复用答案。
Redis 不可用时自动降级为进程内 LRU；**知识库更新或换 embedding 模型后必须清缓存**（导入接口已自动清）。

### 8. 性能：问答为什么慢，以及怎么快起来

排查工具（按顺序跑，几分钟定位瓶颈）：

```powershell
python scripts/profile_query.py    # 把一次问答拆成 12 个阶段逐个计时
python scripts/profile_db.py       # 单独测数据库往返（不加载模型）
python scripts/profile_net.py      # 测网络 RTT 与每次检索的 SQL 条数
python scripts/profile_http.py     # 测真实 HTTP 稳态延迟（问 3 个不同问题）
```

实测结论（本机 RTX 4060 Ti + Neon 新加坡）：

| 阶段 | 耗时 | 性质 |
|---|---|---|
| 问题向量化 | 0.19s | GPU，可忽略 |
| 精排 20 条 | 1.4s | GPU，正常 |
| DeepSeek 生成 | 2.3s | 网络，正常 |
| **每次检索（PGVectorStore 内部十几次往返）** | **2.4~5.4s** | 跨境 SQL |
| **会话读写（4~6 条 SQL）** | **2.6~6.4s** | 跨境 SQL |
| 单条 `SELECT 1` | **0.86s** | 跨境 RTT 0.4s |
| 冷建立连接 | 4.7s | — |

**已做的优化**（50s → 12.5s）：

1. 所有模型走单例（`embed_model_dep()` / `reranker_dep()`）——原本每次问答会重新加载 2.2GB 的 bge-m3 **两遍**，这是最贵的一个 bug；
2. 检索器缓存（`get_hybrid_retriever()`）——原本每请求重建 BM25 索引与融合器，多花约 8s；
3. `SessionStore.add_turn()` 批量写——6 次 SQL 往返压到 4 次；
4. Step-back 查询改写默认关闭（`ENABLE_QUERY_TRANSFORM=false`）——省一次 LLM + 一次完整检索。

**还能更快（按性价比）**：

| 做法 | 预计稳态 | 代价 |
|---|---|---|
| 现状 | **12.5s** | — |
| 会话写入改后台任务（`BackgroundTasks`），响应不等落库 | ~10s | 极端情况可能丢最近一轮历史 |
| **向量与 BM25 语料本地化**（64 块数据用 numpy 暴力余弦，微秒级） | **~6s** | 需要本地索引文件，导入时刷新 |
| **本地 PostgreSQL + pgvector**（架构不变，只改 `DATABASE_URL`） | **~4s** | 需装 Postgres + pgvector（Windows 上 pgvector 要自行编译，建议 Docker） |

> 一句话：**模型不是瓶颈（GPU 上合计约 4 秒），瓶颈是跨境数据库往返。**
> 想进入「3~5 秒」的体感区间，要么把数据搬到本地，要么按上表逐项减少往返次数。



## 五、API 设计

统一响应包装：`{"code": 0, "message": "ok", "data": {...}}`，错误时 `code != 0`。

### POST /api/ingest/upload

`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| file | File | ✅ | PDF/DOCX/HTML/TXT/图片 |
| title | string | ✖ | 不传取文件名 |
| doc_category | string | ✖ | `law` / `plan` / `training` / `general` |
| sync | bool | ✖ | 默认 `true`；大文件建议 `false` 走后台 |

```json
// 200 响应
{"code":0,"message":"索引完成","data":{
  "document":{"id":"uuid","title":"建筑设计防火规范","source_type":"pdf","doc_category":"law",
              "status":"indexed","is_scanned":false,"page_count":120,"chunk_count":486,
              "metadata":{"parser":"pdfplumber"},"created_at":"2025-01-01T10:00:00Z"},
  "chunk_count":486,"elapsed_sec":42.7,"skipped":false}}
```

### DELETE /api/ingest/document/{id}

```json
// 200 响应
{"code":0,"message":"已删除","data":{"document_id":"uuid","deleted_vectors":486}}
// 404：文档不存在
```

### POST /api/query

```json
// 请求
{"question":"甲类厂房与员工宿舍的防火间距是多少？","session_id":null,"user_id":"u_1001",
 "top_k":5,"doc_category":"law","use_cache":true,"use_query_transform":null,"stream":false}
```

```json
// 响应
{"code":0,"message":"ok","data":{
  "answer":"结论：……[1]\n依据：……[2]\n操作要点：……",
  "session_id":"0f2c...","cached":false,"latency_ms":2380,"model":"deepseek-chat",
  "transformed_queries":["甲类厂房与人员密集场所的防火间距设计原则是什么？"],
  "sources":[{"chunk_id":"c1","document_id":"d1","document_title":"建筑设计防火规范",
              "heading_path":"3.4 防火间距","page_no":18,"score":0.91,
              "content":"甲类厂房与……","retrieval_source":"rerank"}]}}
```

### POST /api/eval/retrieval

```json
// 请求
{"dataset_path":"./data/eval/retrieval_golden.jsonl","dataset":[],"top_k":10,
 "run_name":"hybrid+rerank-v2","use_rerank":true}
// dataset 内联格式：[{"question":"...","ground_truth_ids":["chunk-uuid"]}]
```

```json
// 响应
{"code":0,"message":"ok","data":{"run_name":"hybrid+rerank-v2","eval_type":"retrieval",
 "metrics":{"hit_rate@10":0.9,"mrr@10":0.78,"recall@10":0.85,"ndcg@10":0.81},
 "dataset_size":30,"duration_sec":52.4,
 "detail":[{"question":"...","hit":true,"first_rank":1,"retrieved_ids":["c1","c7"]}]}}
```

### POST /api/eval/generation

```json
// 请求
{"dataset_path":"./data/eval/rag_golden.jsonl","dataset":[],
 "metrics":["context_precision","context_recall","faithfulness"],"run_name":"ragas-v1"}
// dataset 内联格式：[{"question":"...","answer":"...","contexts":["..."],"ground_truth":"..."}]
// 缺 answer/contexts 时接口会自动跑一遍问答链路补齐
```

```json
// 响应
{"code":0,"message":"ok","data":{"run_name":"ragas-v1","eval_type":"generation",
 "metrics":{"context_precision":0.82,"context_recall":0.75,"faithfulness":0.94},
 "dataset_size":20,"duration_sec":318.6,"detail":[]}}
```

### GET /health

```json
{"status":"ok","app":"fire-safety-rag","environment":"dev","version":"0.2.0",
 "database":true,"redis":false,"cache_backend":"memory","llm_configured":true,
 "timestamp":"2025-01-01T10:00:00Z"}
```

> `database`/`redis` 是**真实探活结果**：连不上就返回 `false`，服务仍照常启动（接口调用时才报错），
> 这样 Render 不会因为依赖抖动而反复重启。`cache_backend` 为 `memory` 表示语义缓存已降级为内存 LRU。

其余接口：`POST /api/ingest/url`、`POST /api/ingest/text`、`GET /api/ingest/documents`、
`GET /api/ingest/document/{id}/chunks`、`POST /api/ingest/reindex`、
`GET /api/sessions/{id}/history`、`DELETE /api/sessions/{id}`、
`GET /api/eval/records`、`POST /api/eval/build-golden-set`、
`GET /api/admin/stats`、`GET /api/admin/config`、`POST /api/admin/cache/clear`（需 `X-Admin-Token` 头）。

---

## 六、数据库说明（线上 Neon / 本地 Postgres 两种模式）

### 当前用的是哪一种？

**线上 Neon PostgreSQL（ap-southeast-1 新加坡）+ pgvector**，通过 pooler 连接串访问。
本机没有安装任何数据库（`postgres` / `psql` / `docker` 均不存在）。

数据分布：

| 数据 | 位置 |
|---|---|
| `documents` / `chunks` / `data_fire_chunks`（向量）/ `chat_sessions` / `chat_messages` / `eval_records` | ☁️ 线上 Neon |
| `data/uploads`（上传原件）、`data/indexes`（模型与 OCR 缓存）、`data/md`（MinerU 产物） | 💾 本机磁盘 |

**切换数据库只需改 `.env` 的一行 `DATABASE_URL`**，代码里没有硬编码。
`scripts/check_db_modes.py` 可以打印不同连接串下生成的连接参数。

### 两种模式的差异（代码已自动适配）

| | 线上 Neon（pooler） | 本地 Postgres |
|---|---|---|
| 同步参数 | 不下发 `options`（PgBouncer 拒绝启动参数） | 可下发 `-c statement_timeout` |
| `pool_pre_ping` | `True`（Neon 会掐断空闲连接） | `False`（省掉每次取连接的额外往返） |
| 异步 SSL | `ssl=require` | 不强制；要关就写 `?sslmode=disable` |
| 单条 `SELECT 1` | **约 0.86s**（跨境 RTT 0.4s） | 约 1ms |
| 问答稳态 | **约 12.5s** | **约 4s** |

### 用本地 Postgres（开发推荐，能把问答从 12s 压到 4s）

```powershell
# 1) 装 Docker Desktop（一次性，约 1GB）
# 2) 起一个自带 pgvector 的 Postgres（这个镜像已包含 vector 扩展，不用自己编译）
docker run -d --name fsrag-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17

# 3) 改 .env 一行（本地库没开 SSL，要显式 disable，否则会被强制 sslmode=require 拒绝）
#    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable

# 4) 建表并重新导入（换库等于空库，向量要重建）
cd backend
python scripts/ingest_cli.py --file <你的资料> --category law --reset
```

> 为什么不建议 Windows 原生装 Postgres：pgvector 没有官方 Windows 二进制，
> 需要自己用 MSVC 编译，很折腾；Docker 镜像开箱即用。

### 线上 Neon 的注意事项

1. **必须使用 pooler 连接串**：主机名形如 `ep-xxx-pooler.ap-southeast-1.aws.neon.tech`。
   直连串在 Serverless 场景下会快速耗尽连接（Neon 免费套餐连接数很少）。
2. 连接串务必带 `?sslmode=require`。
3. PgBouncer 是 transaction 模式，不能依赖服务端 prepared statement；本项目已在
   `app/db/session.py` 里用 `pool_pre_ping` + 短 `pool_recycle` 适配。
4. ⚠️ **pooler 连接串不允许通过 `options` 传启动参数**（如 `-c statement_timeout=30000`），
   否则报 `unsupported startup parameter in options: statement_timeout`。
   代码里已按 `-pooler` 自动区分。
5. 跑 Alembic 迁移时建议临时切**直连串**（DDL 在 pooler 下偶发不稳），迁完再切回。
6. 业务表（documents/chunks/chat_sessions/chat_messages/eval_records）由 Alembic 或
   `scripts/init_db.sql` 建立；向量表 `data_fire_chunks` 由 PGVectorStore 首次写入时**自动创建**。
7. 换 embedding 模型导致维度变化时，必须 `DROP TABLE data_fire_chunks` 后重建索引。

### 选型建议

| 场景 | 建议 |
|---|---|
| 本地开发 / 调试 | **本地 Postgres**（Docker）：问答 ~4s，迭代快 |
| 部署到 Render / Vercel | **线上 Neon**：容器无状态、免费、免运维 |
| 两边都想用 | 准备两份 `.env`（或环境变量），只换 `DATABASE_URL`，代码不用动 |
| 完全不想装数据库 | 折中方案：向量与 BM25 语料本地化（numpy 暴力检索），Neon 只留业务表 → 约 6s |

---

## 七、评估方法

1. **建立基线的顺序**：先跑 `documented` 数据集（人工标注 20-30 条）→ 记录第一批指标 → 每次只改一个变量。
2. **检索指标看什么**：`MRR` 看排序质量（调 Rerank/融合权重时主要看它），`HitRate/Recall` 看召回能力（调 chunk 大小、HyPE 时看它）。
3. **生成指标看什么**：`faithfulness` 是消防场景的红线（< 0.9 说明在编造），`context_recall` 低说明检索漏了，`context_precision` 低说明上下文噪声大。
4. **评估集自动生成**：`POST /api/eval/build-golden-set` 用 DeepSeek 出题生成草稿，**必须人工校对**后再用于基线。
5. 每次评估的指标与当时的配置快照（模型/参数）都会写入 `eval_records`，便于跨版本对比。

---

## 八、部署

### 后端（Render）

1. 推送代码到 GitHub，Render → New → Blueprint，选择仓库（自动读取 `render.yaml`）。
2. 在环境变量里填 `DEEPSEEK_API_KEY`、`DATABASE_URL`（Neon pooler 串）、`SILICONFLOW_API_KEY`。
3. **免费套餐只有 512MB 内存**，跑不动本地 bge-m3（约 2.2GB）与 reranker（约 2.3GB）：
   线上设 `EMBEDDING_PROVIDER=siliconflow`、`RERANK_PROVIDER=none`，
   或升级到付费套餐后再改回 `local`。
4. 健康检查路径 `/health`，首次冷启动约 30-60 秒属正常。

### 前端（Vercel）

1. Vercel → New Project → 选择仓库，Root Directory 设为 `frontend`。
2. 环境变量 `NEXT_PUBLIC_API_BASE=https://<你的服务>.onrender.com`。
3. 把 Vercel 域名加进后端 `CORS_ORIGINS`。

---

## 九、后续迭代建议

> 📄 **完整的评估策略见 [`docs/评估方案.md`](docs/评估方案.md)**（三份资料版：配额分配、入库前预处理、
> 指标门槛、8 组消融实验、风险清单）。下面只列迭代顺序。

### 第一批收尾（若尚未完成）

- [ ] 用真实资料跑通一条链路：上传 → 检索 → 问答 → 出处正确
- [ ] 人工标注 20-30 条评估集，记录第一批基线指标（MRR/HitRate）
- [x] 管理接口鉴权改为环境变量 `ADMIN_TOKEN`（常时间比较）；**上生产前务必改掉默认值 `change-me`**
- [ ] 问答接口按需加 API Key，防止被刷额度
- [ ] 给 `data/uploads` 与 `data/indexes` 换成对象存储（Render 免费实例无持久磁盘）

### 第二批（本仓库已铺好骨架，按此顺序推进）

| 顺序 | 事项 | 落点 | 验收标准 |
|---|---|---|---|
| 1 | 扫描件 OCR 打通 | `ingestion/ocr.py` + `loader.py` | 一份扫描版规范能完整入库，抽查 10 段文字识别准确率 ≥ 95% |
| 2 | HyPE 生效 | `ingestion/hyphe.py` | HitRate 相对第一批提升 ≥ 5 个百分点 |
| 3 | Query Transformation | `retrieval/query_transform.py` | 口语化提问的 MRR 提升，延迟增量 ≤ 1.5s |
| 4 | 语义缓存 | `utils/cache.py` | 重复/近义提问命中率 ≥ 30%，P95 延迟下降 ≥ 50% |
| 5 | RAGAS 评估 | `evaluation/ragas_eval.py` | faithfulness ≥ 0.9，形成与第一批的对比报告 |
| 6 | LangFuse 可观测 | `main.py` lifespan | 每次问答有 trace，能定位慢请求与幻觉样本 |

### 第三批（可选）

- GraphRAG：把"法规条文 → 引用关系 → 场景"建成知识图谱，回答跨条款的推理型问题
- 多模态：图纸/现场照片识别（PaddleOCR + 视觉大模型）
- 问答反馈闭环：前端 👍/👎 写回评估集，形成持续优化的数据飞轮
- 分级检索：先按 `doc_category` 粗筛再检索，减少跨资料串味

---

## 十、常见问题

| 现象 | 原因与处理 |
|---|---|
| PDF 抽出来是 `FGM(cid:212)T` 这类乱码 | 方正书版字体缺 ToUnicode 映射；`quality.py` 会自动识别并降级 OCR，详见「复杂 PDF 怎么办」 |
| `FileNotFoundError: ...\inference.yml`（PaddleX） | 模型目录只建了目录没下到文件，PaddleX 误以为已存在并跳过下载；删掉该空目录或整个 `%USERPROFILE%\.paddlex` 重跑 |
| `RuntimeError: PDX has already been initialized` | PaddleX 内部重复初始化；设 `PADDLE_PDX_EAGER_INIT=false` 绕过 |
| `unsupported startup parameter in options: statement_timeout` | 用了 Neon pooler 连接串却下发了 `options` 启动参数；本项目已按 `-pooler` 自动规避，自写连接串时删掉 `options` |
| `vector` 扩展创建失败 | Neon 已默认开启；若无权限，在控制台 SQL Editor 手动执行一次 |
| 插入向量报维度错误 | `EMBEDDING_DIM` 与实际模型不符；改配置后 `DROP TABLE data_fire_chunks` 重建 |
| 第一次问答特别慢 | bge-m3/reranker 首次加载；用 `/api/admin/stats` 后续请求作对比，生产建议常驻预热 |
| 问答结果与本地不一致 | 先看 `/api/admin/config` 对比线上参数，再看语义缓存是否命中旧答案 |
| `alembic autogenerate` 生成删除向量表的语句 | 已在 `env.py` 用 `include_object` 排除 `data_*` 表 |
| Render 部署内存溢出 | 关掉本地 embedding 与 rerank，改走 SiliconFlow API |
| `PermissionError ... Check cache directory permissions` | 机器级 `HF_HOME` 指向不可写路径（如盘符根目录下的 `D:\hf-cache`）；本项目用 `MODEL_CACHE_DIR` 并自动降级到可写目录 |
| `PermissionError [WinError 5] ...\AppData\Local\llama_index` | llama-index 默认缓存目录不可写；已通过显式 `cache_folder` + `LLAMA_INDEX_CACHE_DIR` 绕开。可手工删除该目录（需管理员） |
| `No API key found for OpenAI` | LlamaIndex 组件回落到默认 `Settings.llm`（OpenAI）；本项目已用 DeepSeek 注册全局默认 |
| 向量行数远多于切块数 | 历史失败重试留下的孤儿向量；跑 `scripts/db_debug.py` 确认后执行 `ingest_cli.py --reset` 全量重建 |
| 断网后模型加载报 `Unrecognized processing class` | transformers 的 Hub 查询在断网时的误导性报错；已改为优先用本地完整快照，见踩坑 #18 |
| 上传的资料没有 HyPE 假想问题 | 走接口导入时 `index_file()` 需要显式传 `llm`，见踩坑 #19 |

---

## 十一、前端页面

`frontend/` 是 Next.js 单页应用，**本地即可使用，不需要部署**：

```powershell
cd frontend
npm install        # 首次
npm run dev        # → http://localhost:3000
```

四个 Tab：

| Tab | 功能 |
|---|---|
| **问答** | 对话流、`[1]` 引用角标、可展开的出处卡片（条款/页码/相关度/原文）、载入秒数计时、建议问题、新会话 |
| **会话** | 历史会话列表（标题 / 消息数 / 最近预览 / 时间），点开继续对话、删除；按浏览器匿名 ID 隔离 |
| **资料库** | 拖拽或多选上传（带进度条）、资料类别、同步/后台导入、文档状态与错误信息、切块预览（含 HyPE 问题）、删除、重建索引 |
| **设置** | top_k、资料范围过滤、语义缓存开关、Step-back 开关、服务状态与配置、知识库统计、清空语义缓存 |

前端依赖的后端接口（含本轮新增）：

```
GET    /api/sessions                  会话列表（2 条 SQL，避免 N+1）
GET    /api/sessions/{id}/history     会话历史
DELETE /api/sessions/{id}             删除会话
POST   /api/ingest/upload             multipart：file / doc_category / sync
GET    /api/ingest/documents          资料列表
GET    /api/ingest/document/{id}/chunks  切块预览（含 HyPE 问题）
DELETE /api/ingest/document/{id}      删除资料及其向量
POST   /api/ingest/reindex            重建索引
GET    /api/admin/config|stats        服务配置 / 统计（需 X-Admin-Token）
POST   /api/admin/cache/clear         清空语义缓存
```

本地状态都存在浏览器 localStorage（检索参数、管理口令、匿名用户 ID），不会上传。


---

## 十二、踩坑记录（实测）

这轮用真实资料（GB/T 38315—2019，方正书版乱码 PDF）跑通全链路时踩的坑，都已修复：

| # | 现象 | 根因 | 修复位置 |
|---|---|---|---|
| 1 | `unsupported startup parameter in options: statement_timeout` | Neon pooler(PgBouncer) 禁止用 `options` 传启动参数 | `db/session.py` 按 `-pooler` 区分 |
| 2 | PDF 抽出 `FGM(cid:212)T` 乱码，却被当正常文档入库 | 方正书版字体缺 ToUnicode 映射；原判定只看字符数 | 新增 `ingestion/quality.py` + 三级解析阶梯 |
| 3 | `ModuleNotFoundError: fitz` | `requirements.txt` 漏了 PyMuPDF | 已补 `pymupdf` |
| 4 | PaddleOCR 报 unknown argument | 环境里是 3.7，代码按 2.x API 写的 | `ocr.py` 按主版本号适配 |
| 5 | 入库**卡死且无任何输出** | `trust_remote_code=True` 触发交互式确认，非交互环境永久挂起 | `embedder.py` 改为 `False` |
| 6 | `ValueError: invalid literal for int(): 'None'` | 只传 `connection_string` 时 `PGVectorStore` 用全 None 组件拼异步串 | `db/vector_store.py` 显式传 `async_connection_string` + pooler 关闭 prepared statement |
| 7 | `No API key found for OpenAI` | `QueryFusionRetriever` 未拿到 llm 时回落 `Settings.llm`（默认 OpenAI） | `fusion.py` 恒传 DeepSeek + `ensure_llama_settings()` |
| 8 | BM25 语料为空、检索直接崩 | `PGVectorStore` 没有 `get_nodes()`，原实现拿到空列表 | `hybrid.py` 改为直接查 `data_fire_chunks` |
| 9 | `--force` 重建后向量翻倍（64→128） | `node_to_metadata_dict()` 用 `ref_doc_id` **覆盖** metadata 里的 `document_id`（None 序列化成字符串 `'None'`），按它删一行都删不掉 | `indexer.py` 改用 `NodeRelationship.SOURCE`；删除改用 `chunks.id` |
| 10 | reranker 下载失败 `PermissionError at D:\hf-cache` | 机器级 `HF_HOME` 指向盘符根目录，普通用户无写权限 | `config.py` 增加 `HF_HOME` 配置 + 可写性探测兜底 |
| 11 | 有显卡却仍用 CPU | 装的是 CPU 版 torch（`+cpu` 构建不含 CUDA 代码） | 见「用上显卡」；`EMBEDDING_DEVICE=auto` |
| 12 | `.env` 里配了缓存目录却不生效，仍报 `HF_HOME ... 不可写` | **字段名 `hf_home` 与机器级环境变量 `HF_HOME` 同名**，而环境变量优先级高于 .env，配置被静默覆盖 | `config.py` 字段改名 `MODEL_CACHE_DIR`，再由代码强制设置 HF_HOME |
| 13 | `PermissionError: [WinError 5] ... C:\Users\p\AppData\Local\llama_index` | 缓存目录探测失败后返回空值，`HuggingFaceEmbedding` 回落到 `get_cache_dir()`，去创建那个被权限污染的目录 | `apply_hf_cache_env()` 改为**永不返回空**（依次尝试 配置→项目内→系统临时目录），并始终显式传 `cache_folder` |
| 14 | `Detected nested async. Please use nest_asyncio.apply()` / `Event loop is closed` | 处理器写成 `async def`（事件循环里再起循环）；`use_async=True` 还让同步检索依赖临时循环 + asyncpg，第二次检索必然崩 | 路由处理器改同步 `def`（FastAPI 走线程池）+ `retrieval_use_async=False`；命令行脚本不会暴露此问题，只有走 HTTP/ASGI 才复现 |
| 15 | `GET /api/ingest/documents` 报 `Input should be a valid dictionary [input_value=MetaData()]` | Pydantic 字段 `metadata` 与 SQLAlchemy `Base.metadata`（MetaData 对象）**同名冲突**，`model_validate(orm)` 读到的是后者 | `models/document.py` 用 `validation_alias="doc_metadata"` + `serialization_alias="metadata"` |
| 16 | **每次提问都要 50 秒**，连缓存命中也要 21 秒 | ① `utils/cache.py` 用 `build_embed_model()` 而非单例，**每次问答把 2.2GB 的 bge-m3 重新加载两遍**；② 路由每请求重建检索器（含 BM25 索引，+8s）；③ `add_message`×2 在跨国库上是 6 次 SQL 往返 | 全部改用 `model_dep()` 单例（`dependencies.py`）、`get_hybrid_retriever()` 缓存、新增 `SessionStore.add_turn()` 批量写 |
| 17 | 单条 `SELECT 1` 要 0.86 秒 | 数据库在**新加坡 Neon**，跨境 RTT 0.4s，一次问答几十条 SQL → 累积几十秒 | 减少往返（见 #16）；根治要本地库，见「性能」一节 |
| 18 | 断网时报 `Unrecognized processing class in BAAI/bge-m3. Can't instantiate a processor, a tokenizer...`（本地缓存明明是完整的） | 把仓库名交给 transformers 时会先向 Hub 发 HEAD 请求，**断网时走进异常分支**；缓存里再有"下载中断"的残缺快照就更乱 | `embedder.find_cached_model_path()`：先找"配置+权重+分词器"齐全的本地快照，直接把**本地路径**交给加载器，完全不碰 Hub（加载也更快）。reranker 同样处理 |
| 19 | Web 上传的资料 `hypothetical_questions` 为空（HyPE 静默失效），命令行导入却正常 | `routes_ingest.py` 调 `index_file()` 时**没传 `llm`**，而 `attach_hype()` 在没有 LLM 时按设计直接跳过 | 新增 `_hype_llm()`，上传 / 文本 / URL / 重建索引四条路径全部传入 |
| 20 | 双栏 PDF（人大公报版《消防法》）抽出后**左右栏逐行交错**，切块语义破碎 | pdfplumber 默认按 y 坐标逐行输出；两栏间距仅约 30pt，比投影分档还窄 | `loader._looks_two_column()` 用"**词块中心点落在中轴带内的比例**"判定（投影法实测无法区分），命中则先抽顶部跨栏区块、再按左栏→右栏裁剪 |
| 21 | 标准里的表格被压平成断行碎片，数值无法定位（"90 属于哪一列"） | 原实现只用 `extract_text()`，没用 `find_tables()` | `loader._table_to_markdown()`：表格转 Markdown 保留行列，并用 `outside_bbox()` 把表格区域从正文剔除（避免同一内容重复入库） |
| 22 | 归一化后**标题识别全失效**、切块数骤减（64→42） | 全角归一化会去掉"编号与汉字间的空格"（`6.8.5.1 疏散引导行动` → `6.8.5.1疏散引导行动`），而标题正则要求 `\s+` | `chunker.HEADING_PATTERNS` 数字编号后的空格改为可选 `\s*`；GB 51283 段落数因此从 39 恢复到 366 |
| 23 | 库里有表格数值，模型却回答"需查表" | 提示词把每个切块截断到 1200 字符，而防火间距表从第 934 字符才开始，数据行被切掉 | 新增 `PROMPT_NODE_MAX_CHARS=1800`；含 `【表格` 的切块放宽到 3500 字符（`prompt._clip_for_prompt`） |

**两条经验**：

1. **"看起来成功但结果不对"的问题，优先直接查数据库**。第 9 条的 `document_id='None'`
   在任何日志里都看不出来，只有 `SELECT metadata_->>'document_id'` 才暴露。
2. **失败重试会留脏数据**（第 9 条留下 64 条孤儿向量；`documents` 表还留下
   `chunking`/`failed` 状态的记录）。数据对不上时先跑 `scripts/db_debug.py`，
   再用 `ingest_cli.py --reset` 全量重建。

### 精排带来的实际差距（同一问题、同一知识库）

| | 精排关闭 | 精排开启 |
|---|---|---|
| 命中片段 | 术语定义、封面、参考文献（几乎无关） | `6.8.5.1 疏散引导行动`、`6.8.7.2 向消防队提供信息`（精准） |
| 分数 | RRF 0.016（无量纲，无法判断好坏） | reranker 3.24 / 3.00 / 2.90 |
| 答案 | "根据现有资料无法确定" | 逐条引用 `[1][3]` 的完整答案，且未编造 |

结论：**Rerank 是第一批投入产出比最高的一个环节**，别为了省 2.3GB 显存/内存而关掉它。

