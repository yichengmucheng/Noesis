export const PRODUCT_NAME = 'Noesis'
export const PRODUCT_SUBTITLE = '个人知识库'
export const PRODUCT_TAG = ''

export const RETRIEVAL_MODE_OPTIONS = [
  { value: 'hybrid', label: '混合检索', desc: '语义相似与关键词按比例融合' },
  { value: 'mix', label: '知识关联 mix', desc: '文本与跨文档关系一起召回' },
  { value: 'graph', label: '关系路径', desc: '沿实体关系查找跨文档联系' },
  { value: 'vector', label: '语义检索', desc: '只找语义相近的段落' },
  { value: 'keyword', label: '关键词检索', desc: '匹配名称、编号和原文词' },
  { value: 'tree', label: '假设扩写', desc: '先生成假设表述再召回，只在高级检索测试中使用' },
]

export const CHAT_STARTERS = [
  '这份资料里的核心结论是什么？',
  '不同文档对同一件事的说法是否一致？',
  '相关资料之间有哪些联系？',
]

export const RECOMMENDED_RETRIEVAL = {
  retrieval_mode: 'mix',
  top_k: 10,
  similarity_ratio: 0.5,
  similarity_threshold: 0.2,
  rerank_enabled: true,
}

export const STAGE_LABEL: Record<string, string> = {
  uploading: '上传',
  queued: '排队',
  pending: '排队',
  parsing: '解析',
  chunking: '切块',
  embedding: '嵌入',
  graphing: '图谱',
  validating: '校验',
  deleting: '删除',
  succeeded: '完成',
  ready: '完成',
  failed: '失败',
  timeout: '超时',
  cancelled: '已取消',
  consistency_failed: '校验未通过',
}

export const STORE_LABEL: Record<string, string> = {
  doc_status: '文档处理记录',
  full_docs: '文档正文',
  text_chunks: '资料片段',
  full_entities: '知识关联中的实体',
  full_relations: '知识关联中的关系',
  chunks_vdb: '向量索引',
  entities_vdb: '实体索引',
  relationships_vdb: '关系索引',
  graph_nodes: '关系图中的实体',
  graph_edges: '关系图中的连线',
  semantic_cache: '问答缓存',
  qa: '问答记录',
  comparison: '比对记录',
  audit: '审核记录',
  sessions: '会话摘要',
  file_bindings: '文件归属',
  source_files: '原始文件',
  temp_files: '临时文件',
  document_ir: '文档结构',
}
