# 个人知识库 —— 使用与运维手册

> 基于 LightRAG 1.4.9（图谱增强 RAG）。本手册描述的链路已于 2026-09-27 全部实测打通：
> 解析 → 切块 → 实体/关系抽取 → 图谱存储 → 多模式检索 → 生成（带 [KG] 引用）。

---

## 1. 一句话架构

```
你的文件 (.md/.docx/.pdf/.pptx/.xlsx/.txt/.html/.json/.csv ...)
      │ WebUI 上传 / kb_tools.py / 放入 inputs 目录
      ▼
结构化解析（folding式标题层级 / <table> 原子表格 / 中文感知切块）
      ▼
Qwen2.5-7B 实体关系抽取（领域 ontology：设备/部件/故障定义/一~四层原因/措施/效果...）
      ▼
存储：NetworkX 图谱 + NanoVectorDB 向量 + JSON KV   （rag_storage/）
      ▼
检索：naive / local / global / hybrid / mix  →  带 [KG]/[DC] 来源引用的回答
```

## 2. 环境要求

| 组件 | 要求 | 位置 |
|------|------|------|
| Python | 3.12（Anaconda） | `D:\anaconda` |
| LLM | SiliconFlow `Qwen/Qwen2.5-7B-Instruct` | `.env` 中 `LLM_BINDING_*` |
| Embedding | 运行配置中的 `EMBEDDING_MODEL`。当前计划为 `Qwen/Qwen3-Embedding-4B` | `.env` 与 `index_manifest.json` |
| Rerank | 运行配置中的 `RERANK_MODEL`。当前计划为 `Qwen/Qwen3-Reranker-4B` | `.env` 中 `RERANK_*` |
| 密钥 | `.env`（真实密钥，勿入库/勿提交） | 根目录 |

界面上的能力、索引状态和检索诊断显示的是当前进程配置和索引清单里的模型，不使用代码里的固定模型名。嵌入模型、维度、instruction 或切块版本与清单不一致时，系统会停止查询旧向量并要求重建索引。

> ⚠️ 密钥安全：`.env` 含真实 API Key。如需分发，先轮换密钥再用 `env.example` 重新生成。

## 3. 启动 / 停止

```powershell
# 方式一：双击 start_kb.bat 或：
.\start_kb.ps1                      # 需要 http://localhost:9621
# 方式二（手动）：
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
python -u -m lightrag.api.lightrag_server
```

WebUI：http://localhost:9621 ｜ Swagger：http://localhost:9621/docs

停止：在服务窗口按 `Ctrl+C`。

> 已知坑：Windows 控制台默认 GBK，直接 `python -m lightrag.api.lightrag_server` 会在
> 启动横幅处报 `UnicodeEncodeError`（GBK 无法编码 📡）。**必须设置 `PYTHONIOENCODING=utf-8`**
> （start_kb.ps1 已内置）。

### ⚠️ 稳定性红线（2026-09-28 实测踩坑）

1. **绝不要同时跑两个 LightRAG 实例**。它们共用同一个 `rag_storage/`，
   各实例会把自己的内存快照整文件写回磁盘 → 后写者覆盖先写者 → 文档/图谱被抹掉。
2. **绝不要在一个会自动“强杀后台进程”的宿主/会话里跑常驻服务**。
   这种宿主会周期性强杀后台服务器（表现为 `exit code: 1`）；每次强杀可能落在
   LightRAG“读取旧快照 → 尚未持久化”之间，被强杀的实例把不完整快照写回，
   逐步冲掉 doc_status 与 graphml（实测：图谱节点 261→360→再被冲成只剩 1 份文档、图谱文件消失）。
   **本项目的服务请在你自己的终端窗口里跑**，不要交给自动化宿主托管。
3. 若存储已被写坏（`kb_tools.py status` 明显缺文档 / 图谱文件消失），用下方
   *第 11 节重建* 一键恢复。

## 4. 日常使用：kb_tools.py（推荐）

```powershell
$env:PYTHONIOENCODING='utf-8'   # 中文输出需要

python kb_tools.py upload "D:\我的资料\某文档.docx"     # 上传单个文件
python kb_tools.py upload "D:\我的资料"                 # 上传整个目录（递归）
python kb_tools.py scan                                  # 扫描 inputs 目录入库
python kb_tools.py wait                                  # 等处理完成（含失败详情）
python kb_tools.py status                                # 查看文档状态
python kb_tools.py graph                                 # 图谱规模统计
python kb_tools.py ask "节温器打不开怎么判断？" --mode mix  # 问答（默认 mix）
python kb_tools.py benchmark "某问题"                    # 五模式对比评测
python kb_tools.py delete doc-3b4d                       # 按 id 前缀删除文档
python kb_tools.py delete doc-3b4d --file                # 删除时连同源文件
```

## 5. 个人文件导入指南（含飞书）

### 5.1 飞书文档 → 知识库（推荐流程）

1. 飞书文档右上角「更多」→「导出」→ **Word (.docx)**。需要批注就勾选「包含评论」。
2. 导出前自查（影响入库质量）：
   - 标题用飞书**标题样式**（别用加粗冒充标题 → 层级会丢）
   - 复杂合并单元格**先拆分**（否则表格导出成图片，文字丢失）
   - 无版权图片导不出，重要图片改用文字描述
3. 上传：`python kb_tools.py upload "飞书导出文件.docx"`，然后 `python kb_tools.py wait`。
4. 验证：`python kb_tools.py ask "<文档里的知识点>"`，能答出并带来源即入库成功。

> Word 解析已被改造为**结构感知**：Heading 1/2/3… → Markdown `#`/`##`/`###` 层级，
> 表格 → `<table>` 原子块（切块永不从表格中间截断）。见 `lightrag/api/routers/document_routes.py`。

### 5.2 其他格式

| 格式 | 路径 | 说明 |
|------|------|------|
| `.docx` | 结构化解析 + 分层切块 | 标题层级保留；表格原子保护 |
| `.pdf` / `.pptx` | 逐页文本 + `<table>` 感知切块 | 复杂表格在切块中保持完整 |
| `.md` / `.txt` / `.html` / `.json` / `.csv` | 通用 token fallback | 任意长度自动切 |
| `.xlsx` | 工作表逐行转文本 | |
| 扫描件/图片 PDF | 需 DotsOCR（本机缺模型权重与 GPU，暂不可用） | `DOCUMENT_LOADING_ENGINE=DOTSOCR` |

### 5.3 上传方式

- **WebUI 拖拽**（Documents 页 → Upload）
- **CLI**：`kb_tools.py upload`（推荐，中文文件名友好）
- **inputs 目录**：直接把文件丢进 `inputs/`，执行 `kb_tools.py scan`
  （注意：`inputs/__enqueued__/` 是管线内部目录，别手动改）

### 5.4 删除文档

```powershell
python kb_tools.py status                 # 拿到 doc-xxxx 前缀
python kb_tools.py delete doc-xxxx [--file]
```
删除会级联清理：doc_status、全文、chunks、向量、图谱节点/边（基于 LLM 缓存追溯）。

## 6. 检索模式怎么选

| 模式 | 用途 | 验证结论（2026-09-27 实测） |
|------|------|--------------------------|
| naive | 纯向量基线 | 快（~6s），只有 [DC] 文档引用 |
| hybrid | 向量+图谱 | 带出 [KG] 图谱知识 |
| local | 实体邻域 | 图谱事实最丰富；小知识库可能跨案例串扰 |
| global | 关系为主 | 适合跨文档模式归纳 |
| mix | 上面全家桶 | **默认推荐**；最完整的 [KG]+[DC] 引用 |

已知限制：当前知识库仅 9 份文档时，local/global 会出现跨案例污染
（问到 A 案例带出 B 案例内容）。缓解：问答时带上具体上下文（机型/部件名）；
后续按架构文档引入 space/metadata 过滤。

## 7. 数据与备份

```
rag_storage/                     # 全部知识库状态（图谱/向量/文档/KV/缓存）
  graph_chunk_entity_relation.graphml   # 知识图谱（可直接用 Gephi 打开）
  kv_store_*.json / vdb_*.json
inputs/                          # 已入库源文件
lightrag.log                     # 运行日志
.env                             # API 密钥（私密）
```

迁移到新机器 = 拷贝整个 `LightRAG_test` 目录 + 一个能访问 SiliconFlow 的网络。
备份知识库 = 备份 `rag_storage/` + `inputs/` 两个目录；同时必须备份
`product_app.sqlite`（会话、消息、反馈和已确认个人记忆）。应用库启用 WAL，
不要只复制正在运行中的 `-wal` / `-shm` 文件，使用 SQLite backup 命令生成一致性快照：

```powershell
python -m lightrag.product_appdb_backup --working-dir data/rag_storage `
  --backup backups/product_app-$(Get-Date -Format yyyyMMddHHmmss).sqlite
python -m lightrag.product_appdb_backup --working-dir data/rag_storage --check
```

恢复前停止 API 和 Worker，确认备份来自同一用户数据快照，然后执行：

```powershell
python -m lightrag.product_appdb_backup --working-dir data/rag_storage `
  --backup backups/product_app-20261002.sqlite --restore --check
```

恢复后重新启动服务，并检查会话、反馈、记忆和知识库归属。应用库当前支持单机
SQLite + 一个 API 进程及一个 Worker 进程；多 API 进程部署前应迁移到 PostgreSQL，
本阶段不自动引入 PostgreSQL。

## 8. 半双工语音练习（Phase C1）

语音练习是浏览器录音 → ASR 转写 → 复用可信问答 → TTS 播放的半双工流程。语音不会绕过
知识库检索、文档引用或个人记忆作用域。练习会话和回合保存在 `product_app.sqlite`，随知识库
删除一起清理，并随应用库备份恢复。

需要外部 OpenAI-compatible 服务时，在部署环境设置 `ASR_API_BASE`、`ASR_API_KEY`、
`ASR_MODEL` 与 `TTS_API_BASE`、`TTS_API_KEY`、`TTS_MODEL`、`TTS_VOICE`。未配置时，界面会
明确提示服务尚未配置，不会生成占位文本或空音频。第一版不做全双工抢话、声纹、发音评分、
情绪识别、视频或音频资料入库。

API：`POST /api/v1/voice/practice/sessions`、`POST /api/v1/voice/transcribe`、
`POST /api/v1/voice/practice/{session_id}/turn`、`POST /api/v1/voice/speech`、
`POST /api/v1/voice/practice/{session_id}/finish`。所有接口都要求当前用户身份，练习回合必须
属于当前知识库。

## 9. 测试

```powershell
python -m pytest -q     # 后端全量（含 8 个新增原生切块单测）
# 预期：27 passed；1 个 test_generate_with_system 失败属环境预存问题
#（该测试需要本机 11434 端口的 Ollama 服务，与 LightRAG 主链路无关）
```

## 10. 本次改造的技术细节（2026-09-27）

1. **docx 解析结构化**（`document_routes.py`）：原实现把段落拼成一串、表格变制表符文本；
   现按文档顺序遍历段落+表格，Heading 样式→`#` 层级，表格→`<table>` 原子块。
2. **docx 切块重写**（`operate.py`）：删除致命的 `from langchain.text_splitter import ...`
   （langchain 1.x 无此模块，Word 入库必崩）；换成原生实现：
   标题层级切块 + 中文分隔符（空行→换行→句读）智能二切 + 表格原子保护 + overlap 硬切兜底。
   附 8 个单元测试 `tests/test_native_chunking.py`。
3. **失败路径修复**（`lightrag.py`）：`processing_start_time` 提前初始化，
   切块抛异常时不再触发 `UnboundLocalError` 二次崩溃。
4. **kb_tools.py**：绕过系统代理（trust_env=False，防 Clash 7897 拦本地请求）、
   轮询容错（服务处理期丢弃 keep-alive 的瞬时 10053/10054）。
5. **启动脚本** `start_kb.ps1/.bat`：内置 UTF-8 环境、端口探测、目录校验。

## 11. 常见问题（FAQ）

**Q: upload 返回 502 / connection reset？**
A: Python `httpx` 默认读系统注册表代理（本机 Clash 127.0.0.1:7897），
本地请求被代理拦截。kb_tools 已强制直连；如自己写脚本，加 `trust_env=False`。

**Q: 上传报 duplicated？**
A: 同名文件已在 inputs/ 里。先 `kb_tools.py status`，用 `delete` 清理或改名再传。

**Q: 处理卡在 processing 很久？**
A: 正常。MAX_ASYNC=1 + 7B 模型，单文档约 1-3 分钟。看 `lightrag.log` 尾部。

**Q: WebUI 打不开？**
A: 确认 start_kb.ps1 无报错且端口 9621 监听；浏览器走系统代理时把 localhost 加入直连
（ProxyOverride 默认已含）。

**Q: 想换更强的模型？**
A: 编辑 `.env` 的 `LLM_MODEL`（如 `Qwen/Qwen2.5-32B-Instruct` 或 Qwen3 系列），
重启服务。Embedding 不要动（换维度需重建索引）。

## 12. 存储损坏后的重建（一键修复）

源文档都保存在 `inputs/` 与 `lightrag/api/routers/output/*.md`（OCR 抽取结果），
即使 `rag_storage` 被写坏也随时可重建：

```powershell
# 完整重建（把现有 rag_storage 改名备份，从空目录重灌全部源文档）
.\rebuild_kb.ps1 -Reset

# 不动现有存储，只追加缺失文档（推荐日常用）
.\rebuild_kb.ps1
```

脚本自动：备份旧存储 → 启动独立服务器 → 上传 `output/*.md`（6 份故障报告 OCR +
个人文档）+ 飞书样例 docx → 等待全部处理 → 打印最终状态。
要重建后常驻服务供 WebUI 使用，加 `-Keep`。
重建全程约 10–15 分钟（单文档 1–3 分钟），期间保持窗口打开。

## 13. 账户、迁移、备份和删除

开发环境保持 `PRODUCT_AUTH` 为空。控制台可以匿名使用，已有知识库归在迁移用户 `local-owner` 下。

开启登录前，先注册目标邮箱，再把旧库交过去。迁移可以重复执行，第二次不会再次改写：

```powershell
python -m lightrag.product_migrate --working-dir data/rag_storage --email you@example.com
```

也可以在 `.env` 设置 `PRODUCT_OWNER_EMAIL=you@example.com`。该邮箱登录或注册时，会把仍属于 `local-owner` 的知识库转给这个用户。文档绑定、问答和缓存随知识库一起转移。没有 `file_path` 或 `kb_id` 的旧图谱节点和旧边不会返回给任何用户，需要重新入库后才会出现。

检查某个库删除后是否还有残留。这条命令会解析文档状态、全文、文本块、向量库、NetworkX 图、缓存、问答、审计和源文件，不会只看文件在不在：

```powershell
python -m lightrag.product_migrate --working-dir data/rag_storage --input-dir data/originals --deep-check-kb <kb_id>
```

新上传的文件按 `owner_id/kb_id/doc_id/文件名` 存放。不同用户、不同知识库可以有相同显示名。同一知识库再次上传相同显示名会生成新版本。

接口 `GET /api/v1/kb/purge-jobs/{job_id}/check` 返回同样的残留清单。删除任务写在 `product_shell.json`，服务重启后会继续未完成的任务。另一个知识库仍绑定的文件不会被删掉。

生产启动：

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --force-recreate
```

`.env` 里要有独立的 `TOKEN_SECRET`、`PRODUCT_AUTH=1`、明确的 `CORS_ORIGINS`。HTTPS 部署再把 `COOKIE_SECURE=1`。`APP_ENV=production` 时如果登录关闭、令牌密钥仍是默认值，或 CORS 为 `*`，进程拒绝启动。

修改 `.env` 后必须 `--force-recreate`。`docker restart` 不会重新读取环境变量。

备份时复制 `data/rag_storage`、`data/originals`，并使用上面的 SQLite backup 命令保存
`product_app.sqlite`。密钥轮换时先在模型平台更换密钥，再写入 `.env` 并重建容器，
不要把 `.env` 提交到仓库。刷新令牌放在 HttpOnly Cookie 中，访问令牌只留在浏览器内存。

## 14. 任务队列

上传和删除不再放在接口进程的内存里。接口只创建任务并返回 `job_id`，独立进程执行：

```powershell
python -m lightrag.product_worker
```

开发环境不设置 `DATABASE_URL` 时，任务写在 `WORKING_DIR/product_jobs.sqlite`。生产环境的 Worker 必须配置 `DATABASE_URL=postgresql://...`，密码由部署环境注入，不要写进仓库。`docker compose` 会同时启动 `lightrag` 和 `lightrag-worker`。

任务被租约锁定。Worker 退出时释放租约；进程崩溃后，租约到期的任务回到队列，由其他 Worker 接管。同一个幂等键不会创建第二份任务。文档只有一致性检查通过后才会变成可检索。

检查存储和过期租约：

```powershell
python -m lightrag.product_storage_check --working-dir data/rag_storage
python -m lightrag.product_storage_check --working-dir data/rag_storage --kb-id <kb_id>
python -m lightrag.product_storage_check --working-dir data/rag_storage --repair-safe
```

`--repair-safe` 只恢复已经存在且可读的备份，或把过期租约收回队列。它不会猜测 `owner_id` 或 `kb_id`。无法确认归属的数据留在隔离区，不参与检索。
