# Noesis

图谱增强的通用知识推理系统 —— 把任意领域文档变成可推理的知识网络。

本仓库基于 [LightRAG](https://github.com/HKUDS/LightRAG) 进行领域定制与工程落地，当前主线场景为**故障诊断定位问答**（发动机 A3 分析报告），技术框架可横向扩展到质量管理、工艺规程、标准法规等领域。

> 上游完整文档见 [`README.LightRAG.md`](./README.LightRAG.md) / [`README-zh.md`](./README-zh.md)  
> 业务汇报材料见 [`docs/故障诊断智能体项目介绍.md`](./docs/故障诊断智能体项目介绍.md)

---

## 我们解决什么问题

工程现场排查故障时，常见痛点是：

- 历史 PDF / Word 报告数量大，人工翻找成本高
- 关键词搜索找不到文档深处的**因果关系**
- 经验集中在个人，难以传承

Noesis 的目标：用自然语言提问，在数秒内返回**有依据、可追溯来源**的答案，并可视化知识关联网络。

---

## 整体框架

```
┌─────────────────────────────────────────────────────────────┐
│                     WebUI / REST API                          │
│                   http://localhost:9621                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                 LightRAG Server (FastAPI)                     │
│  文档管理 · 图谱增强检索 · 回答生成 · 知识图谱可视化          │
└───────┬─────────────────────┬─────────────────┬─────────────┘
        │                     │                 │
   ┌────▼────┐          ┌─────▼─────┐     ┌─────▼─────┐
   │  LLM    │          │ Embedding │     │  Storage  │
   │ 理解/生成│          │ 语义向量  │     │ KV/向量/图│
   └─────────┘          └───────────┘     └───────────┘

可选 OCR 子系统（dots.ocr，默认端口 7797）
PDF/扫描件/复杂表格 → 结构化 Markdown → 再入库
```

### 三步工作流

1. **知识入库**  
   文档上传 → 文本抽取（PDF/DOCX，或 OCR）→ Token 分块 → LLM 抽取实体与关系 → 写入向量库 + 知识图谱

2. **智能检索**  
   用户提问 → 同时做语义向量检索 + 图谱关联检索（local / global / hybrid / mix 等模式）→ 召回相关片段与因果链

3. **生成回答**  
   融合多来源上下文 → LLM 生成中文答案 → 附带来源引用

与普通 RAG 的核心差异：**多了一层知识图谱**，既能找“相似”，也能沿因果链追溯。

---

## 关键代码位置

| 能力 | 路径 | 说明 |
|------|------|------|
| 服务入口 | `lightrag/api/lightrag_server.py` | FastAPI + WebUI，默认端口 `9621` |
| 核心管线 | `lightrag/lightrag.py` | 入库 `ainsert`、查询 `aquery` |
| 抽取 / 检索 | `lightrag/operate.py` | 实体关系抽取、`kg_query`、分块 |
| **领域 Prompt** | `lightrag/prompt.py` | 故障知识图谱专家角色、A3 结构抽取规则、中文回答模板 |
| **实体类型** | `lightrag/constants.py` | 设备/部件/故障/多层原因/措施/验证等 |
| 文档 API + OCR 钩子 | `lightrag/api/routers/document_routes.py` | 上传入库；`process_with_ocr()` 对接 OCR 服务 |
| 查询 API | `lightrag/api/routers/query_routes.py` | `/query`、流式查询 |
| 前端 | `lightrag_webui/`（构建产物在 `lightrag/api/webui/`） | 文档 / 图谱 / 检索界面 |
| OCR 服务 | `dots.ocr/` | 独立 FastAPI，`/api/parse-file`，端口 `7797` |

### 本仓库相对上游的定制点

- **中文 + 故障域实体类型**：`DEFAULT_SUMMARY_LANGUAGE = "Chinese"`，实体覆盖失效网多层原因、措施、验证、效果等
- **A3 报告 Prompt Engineering**：按失效网 / 根本原因 / 措施表 / HTML 跨行表格规则抽取
- **OCR 联调能力**：`document_routes.py` 中预留 DotsOCR 调用（需 OCR 服务在线；当前默认入库仍可走本地 PDF/DOCX 解析）
- **中文文件名排序**等工程体验优化

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- 可用的 LLM API（OpenAI 兼容接口即可）与 Embedding 服务

### 2. 安装

```bash
git clone https://github.com/yichengmucheng/Noesis.git
cd Noesis
pip install -e ".[api]"
```

### 3. 配置

```bash
cp env.example .env
# 编辑 .env：填写 LLM_BINDING_*、EMBEDDING_BINDING_* 等
```

关键配置项：

| 变量 | 含义 |
|------|------|
| `HOST` / `PORT` | 服务监听，默认 `0.0.0.0:9621` |
| `SUMMARY_LANGUAGE` | 建议 `Chinese` |
| `LLM_BINDING` / `LLM_MODEL` / `LLM_BINDING_HOST` / `LLM_BINDING_API_KEY` | 大模型 |
| `EMBEDDING_BINDING` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` / ... | 向量模型 |
| `LIGHTRAG_GRAPH_STORAGE` | 图存储，本地可用 NetworkX |

### 4. 启动

```bash
lightrag-server
# 或：python -m lightrag.api.lightrag_server
```

浏览器打开：<http://localhost:9621>

### 5.（可选）启动 OCR

```bash
cd dots.ocr
# 按 dots.ocr/README.md 准备模型后
python -m dots_ocr.main
# 默认 http://0.0.0.0:7797
```

Docker 可选方案见根目录 `docker-compose.yml`。

---

## 目录结构（精简）

```
Noesis/
├── lightrag/              # 核心库与 API
├── lightrag_webui/        # 前端源码
├── dots.ocr/              # 可选 OCR 子系统
├── examples/              # 示例脚本
├── docs/                  # 部署与项目说明
├── env.example            # 配置模板（勿提交真实 .env）
├── docker-compose.yml
└── README.md              # 本文件
```

运行时数据（默认不入库）：

- `inputs/`：上传文档
- `rag_storage/`：向量、KV、图谱存储

---

## 演示建议

1. 文档页：上传若干领域报告  
2. 知识图谱页：查看自动构建的实体关系网络  
3. 检索页：用自然语言提问，检查答案与来源引用  

示例问题（故障场景）：

> K08N 机型 EGR 冷却器连接胶管漏水的根本原因是什么？有哪些改进措施？

---

## 致谢

- 检索与图谱增强框架：[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- OCR 能力参考仓库内 `dots.ocr` 及相关多模态解析方案

## License

遵循本仓库 `LICENSE` 及上游 LightRAG / dots.ocr 各自许可协议。
