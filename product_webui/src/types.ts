export interface KnowledgeBase {
  id: string
  name: string
  description?: string
  created_at?: string
  updated_at?: string
  owner_id?: string
}

export interface Document {
  id: string
  name: string
  chunk_count?: number
  char_count?: number
  file_size?: number
  updated_at?: string
  created_at?: string
  status?: string
  progress?: number
  error_msg?: string
  file_path?: string
  job_id?: string
  mime_type?: string
  version_id?: string
  unit_count?: number
  index_status?: string
  version?: string
}

export interface ProductJob {
  job_id: string
  job_type?: string
  user_id?: string
  kb_id?: string
  doc_id?: string
  file_name?: string
  status?: string
  stage?: string
  step?: string
  progress?: number
  attempt?: number
  max_attempts?: number
  error_code?: string
  error_message?: string
  error?: string
  created_at?: string
  started_at?: string
  finished_at?: string
  retry_at?: string
}

export interface PurgeJob extends ProductJob {
  consistency_check_id?: string
}

export interface DeviceSession {
  session_id: string
  device_label?: string
  created_at?: string
  last_used_at?: string
  expires_at?: string
  is_current?: boolean
}

export interface QueryPlan {
  route?: string
  standalone_query?: string
  reason?: string
  variants?: string[]
  hyde_text?: string
  subquestions?: string[]
  llm_used?: boolean
  fallback_reason?: string
}

export interface GraphPath {
  path_id?: string
  path?: string
  hop?: number
  source_id?: string
  parent_id?: string
}

export interface SearchResult extends GraphPath {
  chunk_id?: string
  document_id?: string
  doc_name?: string
  unit_id?: string
  excerpt?: string
  content?: string
  score?: number
  raw_score?: number
  ranking_score?: number
  admission_score?: number
  admission_reason?: string
  channel?: string
  channel_rank?: number
  query_variant_id?: string
  source?: string
  page_number?: number
  page_num?: number
  slide_number?: number
  section_path?: string[]
  sheet_name?: string
  cell_range?: string
  parent_content?: string
  reranked?: boolean
}

export interface Citation {
  citation_id?: string
  doc_name?: string
  document_id?: string
  version_id?: string
  chunk_id?: string
  unit_id?: string
  evidence_ref?: Record<string, unknown>
  excerpt?: string
  page_number?: number
  page_num?: number
  slide_number?: number
  section_path?: string[]
  sheet_name?: string
  cell_range?: string
  bbox?: number[] | null
  line_start?: number | null
  line_end?: number | null
  location?: {
    page_number?: number | null
    slide_number?: number | null
    section_path?: string[]
    sheet_name?: string
    cell_range?: string
    bbox?: number[] | null
    line_start?: number | null
    line_end?: number | null
  }
  parent_id?: string
  parent_content?: string
}

export interface SourceContent {
  source_kind?: string
  text?: string
  units?: Array<Record<string, unknown>>
  layout?: string
  notice?: string
  doc_name?: string
  document_id?: string
  version_id?: string
}

export interface DocumentSource {
  document_id?: string
  version_id?: string
  doc_name?: string
  mime_type?: string
  source_kind?: string
  unit?: Citation['location'] & { unit_id?: string }
  matched_chunk?: { chunk_id?: string; excerpt?: string }
  parent_context?: string
  source_text?: string
  preview?: { available?: boolean; kind?: string; url?: string | null }
}

export interface IndexStatus {
  status: 'ready' | 'index_rebuild_required' | string
  reason?: string
  embedding_model?: string
  embedding_dimension?: number | null
  chunking_version?: string
  parent_chunk_version?: string
  corpus_revision?: string
  created_at?: string
  rerank_model?: string
}

export interface ChatEvent {
  type: 'meta' | 'token' | 'error' | 'done' | string
  text?: string
  message?: string
  citations?: Citation[]
  answer?: string
  answerable?: boolean
  degraded?: boolean
  conversation_id?: string
}

export interface ConversationItem {
  id: string
  title?: string
  summary?: string
  is_pinned?: boolean
  is_archived?: boolean
  updated_at?: string
  last_message_at?: string
}

export interface StoredMessage {
  id: string
  conversation_id?: string
  role: 'user' | 'assistant' | string
  content: string
  status?: string
  citations?: Citation[]
  created_at?: string
  retrieval_meta?: { memories?: { marker?: string; content?: string }[] }
}

export interface ProductCapabilities {
  max_upload_bytes: number
  max_files_per_kb: number
  max_pdf_pages?: number
  extensions: string[]
  embedding_model?: string
  rerank_model?: string
  llm_model?: string
  embedding_dimension?: number
}

export interface RetrievalCounts {
  dense_top?: number
  bm25_top?: number
  graph_top?: number
  rrf?: number
  rerank_pool?: number
  children?: number
  parents?: number
}

export interface SearchResponse {
  status?: string
  query?: string
  empty_reason?: string
  rebuild_reason?: string
  query_plan?: QueryPlan | null
  retrieval?: RetrievalCounts
  answer_context?: { parent_id?: string; content?: string; chunk_ids?: string[] }[]
  chunks?: SearchResult[]
  hyde_text?: string | null
  timings?: Record<string, number>
  rerank?: { enabled?: boolean; applied?: boolean; model?: string; warning?: string | null; pool_size?: number; reranked?: number }
  index_version?: string
  embedding_model?: string
  rerank_model?: string
}

export interface JobPage {
  items: ProductJob[]
  total: number
  page: number
  page_size: number
}

export interface DeepCheckStore {
  count: number
  samples?: string[]
}

export interface DeepCheckReport {
  passed?: boolean
  stores?: Record<string, DeepCheckStore>
  orphan_count?: number
  job?: ProductJob
}

export interface KbSettings {
  retrieval_mode?: string
  top_k?: number
  similarity_ratio?: number
  similarity_threshold?: number
  rerank_enabled?: boolean
  graph_enabled?: boolean
  context_expand?: boolean
  answer_detail?: 'concise' | 'standard' | 'detailed' | 'brief' | 'normal'
  show_citations?: boolean
  ignore_whitespace?: boolean
  only_changes?: boolean
  mark_number_changes?: boolean
  audit_auto_pass?: boolean
  audit_reject_comment?: boolean
}

export interface MemoryCandidate {
  id: string
  content: string
  category?: string
  status?: string
  kb_id?: string
}

export interface MemoryItem {
  id: string
  content: string
  category?: string
  enabled?: boolean
  source_type?: string
  scope?: 'global' | 'kb' | string
  kb_id?: string
  expires_at?: string
  source_status?: string
  created_at?: string
  updated_at?: string
  memory_score?: number
  retrieval_reason?: string
}

export interface VoicePracticeSession {
  id: string
  owner_id?: string
  kb_id: string
  goal?: 'free' | 'recall' | 'interview' | 'review' | string
  status?: 'active' | 'completed' | 'cancelled' | 'failed' | string
  conversation_id?: string
  summary?: string
  created_at?: string
  updated_at?: string
  ended_at?: string
  turn_count?: number
  completed_turn_count?: number
  cited_turn_count?: number
  first_transcript?: string
  turns?: VoicePracticeTurn[]
}

export interface VoicePracticeTurn {
  id: string
  session_id: string
  kb_id: string
  transcript: string
  answer?: string
  citations?: Citation[]
  memory_refs?: { marker?: string; content?: string; memory_id?: string }[]
  audio_status?: string
  status?: string
  created_at?: string
}

export interface SourceReaderProps {
  documentId?: string
  citation: Citation
  onClose: () => void
}
