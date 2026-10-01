# Noesis

个人知识库。把资料整理成可检索、可核对来源的知识网络。

界面入口是 `product_webui`，服务在 `lightrag/api`。底层检索基于 [LightRAG](README-lightrag.md)。

## 本地运行

1. 复制 `.env.example` 为 `.env`，填入自己的模型地址。不要把 `.env` 提交到仓库。
2. 安装 Python 依赖后启动 API。
3. 在 `product_webui` 执行 `npm install` 和 `npm run build`，由 API 托管 `dist`。

开发环境可以不开启登录。生产环境必须开启账户，并单独设置 `TOKEN_SECRET`。

## 目录

- `product_webui`：资料、问答、账户和高级检索测试
- `lightrag`：解析、索引、检索和任务
- `tests`：单元测试和 Playwright
