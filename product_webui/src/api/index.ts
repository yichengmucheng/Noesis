import axios from 'axios'
import type {
  DeepCheckReport, DeviceSession, DocumentSource, IndexStatus, JobPage, KbSettings,
  KnowledgeBase, ProductCapabilities, ProductJob, PurgeJob, SearchResponse,
} from '../types'

localStorage.removeItem('kb-refresh')

const http = axios.create({
  baseURL: '/api/v1',
  timeout: 180000,
  withCredentials: true,
  headers: { 'X-KB-Request': '1' },
})

let accessToken = ''
let refreshPromise: Promise<boolean> | null = null

export function authHeader(): Record<string, string> {
  return accessToken ? { Authorization: `Bearer ${accessToken}` } : {}
}

export function isLoggedIn() {
  return Boolean(accessToken)
}

export function setTokens(access: string) {
  accessToken = access || ''
}

export function clearTokens() {
  accessToken = ''
}

async function refreshOnce() {
  try {
    const response = await axios.post('/api/v1/auth/refresh', {}, {
      withCredentials: true,
      timeout: 8000,
      headers: { 'X-KB-Request': '1' },
    })
    accessToken = response.data.access_token || ''
    return Boolean(accessToken)
  } catch {
    accessToken = ''
    return false
  }
}

export function forceRefresh() {
  if (!refreshPromise) {
    refreshPromise = refreshOnce().finally(() => { refreshPromise = null })
  }
  return refreshPromise
}

export function ensureSession() {
  if (accessToken) return Promise.resolve(true)
  return forceRefresh()
}

function loginPath() {
  const base = import.meta.env.BASE_URL || '/'
  return `${base}login`.replace(/([^:]\/)\/+/g, '$1')
}

http.interceptors.request.use((config) => {
  if (accessToken) config.headers.Authorization = `Bearer ${accessToken}`
  return config
})

http.interceptors.response.use(
  (r) => r.data,
  async (err) => {
    const config = err?.config
    const url = String(config?.url || '')
    if (err?.response?.status === 401 && config && !config._retry && !url.includes('/auth/')) {
      config._retry = true
      const ok = await forceRefresh()
      if (ok) {
        config.headers.Authorization = `Bearer ${accessToken}`
        return http(config)
      }
      if (!window.location.pathname.endsWith('/login')) window.location.assign(loginPath())
    }
    return Promise.reject(err?.response?.data?.detail ?? err.message)
  },
)

export const authApi = {
  status: () => http.get<any, any>('/auth/status'),
  register: (email: string, password: string) =>
    http.post<any, any>('/auth/register', {
      email,
      password,
      device_label: navigator.userAgent.slice(0, 80),
    }),
  login: (email: string, password: string) =>
    http.post<any, any>('/auth/login', {
      email,
      password,
      device_label: navigator.userAgent.slice(0, 80),
    }),
  logout: () => http.post<any, any>('/auth/logout', {}),
  logoutAll: () => http.post<any, any>('/auth/logout-all', {}),
  changePassword: (oldPassword: string, newPassword: string) =>
    http.post<any, any>('/auth/password', { old_password: oldPassword, new_password: newPassword }),
  me: () => http.get<any, { email?: string; user_id?: string }>('/auth/me'),
  sessions: () => http.get<any, { items: DeviceSession[] }>('/auth/sessions'),
  revokeSession: (sessionId: string) => http.delete<any, { status: string }>(`/auth/sessions/${sessionId}`),
}

// ── 知识库 ────────────────────────────────────────────────────
export const kbApi = {
  list: (tenantId = 'default') =>
    http.get<any, { items: KnowledgeBase[] }>('/kb', { params: { tenant_id: tenantId } }),

  capabilities: () => http.get<any, ProductCapabilities>('/capabilities'),

  create: (data: { name: string; description?: string }) =>
    http.post<any, any>('/kb', data),

  update: (kbId: string, data: { name: string; description?: string }) =>
    http.put<any, any>(`/kb/${kbId}`, data),

  get: (kbId: string) =>
    http.get<any, any>(`/kb/${kbId}`),

  delete: (kbId: string) =>
    http.delete<any, any>(`/kb/${kbId}`),

  purgeJob: (jobId: string) =>
    http.get<any, PurgeJob>(`/kb/purge-jobs/${jobId}`),

  retryPurge: (jobId: string) =>
    http.post<any, PurgeJob>(`/kb/purge-jobs/${jobId}/retry`),

  checkPurge: (jobId: string) =>
    http.get<any, DeepCheckReport>(`/kb/purge-jobs/${jobId}/check`),

  deepCheck: (jobId: string) =>
    http.get<any, DeepCheckReport>(`/kb/purge-jobs/${jobId}/deep-check`),

  jobs: (kbId: string, params: { page?: number; page_size?: number; status?: string } = {}) =>
    http.get<any, JobPage>(`/kb/${kbId}/jobs`, { params }),

  indexStatus: (kbId: string) =>
    http.get<any, IndexStatus>(`/kb/${kbId}/index-status`),

  rebuildIndex: (kbId: string) =>
    http.post<any, ProductJob>(`/kb/${kbId}/index-rebuild`, {}),

  getSettings: (kbId: string) =>
    http.get<any, KbSettings>(`/kb/${kbId}/settings`),

  updateSettings: (kbId: string, data: KbSettings) =>
    http.put<any, KbSettings>(`/kb/${kbId}/settings`, data),
}

// ── 文档 ──────────────────────────────────────────────────────
export const docApi = {
  list: (params: { kb_id: string; page?: number; page_size?: number; search?: string }) =>
    http.get<any, any>('/documents', { params }),

  upload: (formData: FormData) =>
    http.post<any, any>('/documents', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  getStatus: (docId: string) =>
    http.get<any, any>(`/documents/${docId}/status`),

  delete: (docId: string, kbId: string) =>
    http.delete<any, any>(`/documents/${docId}`, { params: { kb_id: kbId } }),

  source: (docId: string, params: { kb_id: string; version_id?: string; chunk_id?: string; unit_id?: string }) =>
    http.get<any, DocumentSource>(`/documents/${docId}/source`, { params }),

  cancel: (kbId: string, name: string, docId = '') =>
    http.post<any, any>('/documents/cancel', null, { params: { kb_id: kbId, name, doc_id: docId } }),

  chunks: (params: { kb_id: string; page?: number; page_size?: number; search?: string }) =>
    http.get<any, any>('/chunks', { params }),
}

export const jobApi = {
  get: (jobId: string) => http.get<any, ProductJob>(`/jobs/${jobId}`),
  list: (params: { kb_id?: string; doc_id?: string; status?: string; page?: number; page_size?: number } = {}) =>
    http.get<any, JobPage>('/jobs', { params }),
  cancel: (jobId: string) => http.post<any, ProductJob>(`/jobs/${jobId}/cancel`, {}),
  retry: (jobId: string) => http.post<any, ProductJob>(`/jobs/${jobId}/retry`, {}),
}

export const qaApi = {
  list: (params: { kb_id: string; page?: number; page_size?: number; search?: string }) =>
    http.get<any, any>('/qa', { params }),

  create: (data: { kb_id: string; question: string; answer: string; doc_id?: string; doc_name?: string }) =>
    http.post<any, any>('/qa', data),

  delete: (qaId: string, kbId: string) =>
    http.delete<any, any>(`/qa/${qaId}`, { params: { kb_id: kbId } }),
}

// ── 检索 & 问答 ───────────────────────────────────────────────
export const chatApi = {
  searchTest: (data: {
    query: string
    kb_id: string
    retrieval_mode?: string
    top_k?: number
    similarity_ratio?: number
    enable_rerank?: boolean
    similarity_threshold?: number
    history?: { role: string; content: string }[]
  }) => http.post<any, SearchResponse>('/chat/search-test', data),

  chat: (data: {
    query: string
    kb_id: string
    retrieval_mode?: string
    stream?: boolean
    history?: { role: string; content: string }[]
  }) => http.post<any, any>('/chat', data),
}

// ── 知识图谱 ──────────────────────────────────────────────────
export const graphApi = {
  getStats: (kbId: string) =>
    http.get<any, any>('/graph/stats', { params: { kb_id: kbId } }),

  getEntities: (kbId: string, params?: {
    entity_type?: string; search?: string; page?: number; page_size?: number; sort?: string
  }) =>
    http.get<any, any>('/graph/entities', { params: { kb_id: kbId, ...params } }),

  getEntityNeighbors: (kbId: string, entityName: string, hops = 2) =>
    http.get<any, any>(`/graph/entities/${encodeURIComponent(entityName)}/neighbors`, {
      params: { kb_id: kbId, hops },
    }),

  getRelations: (kbId: string, params?: {
    entity_name?: string; rel_type?: string; page?: number; page_size?: number
  }) =>
    http.get<any, any>('/graph/relations', { params: { kb_id: kbId, ...params } }),

  getSubgraph: (kbId: string, entityName?: string, hops = 2, limit = 100) =>
    http.get<any, any>('/graph/subgraph', {
      params: { kb_id: kbId, entity_name: entityName, hops, limit },
    }),

  getConfig: (kbId: string) =>
    http.get<any, any>('/graph/config', { params: { kb_id: kbId } }),

  updateConfig: (kbId: string, data: any) =>
    http.put<any, any>('/graph/config', data, { params: { kb_id: kbId } }),
}

// ── 知识比对 ──────────────────────────────────────────────────
export const comparisonApi = {
  list: (kbId: string) =>
    http.get<any, any>('/kb/comparison', { params: { kb_id: kbId } }),

  create: (data: { kb_id: string; doc_id_old: string; doc_id_new: string; reason?: string; compare_type?: string }) =>
    http.post<any, any>('/kb/comparison', data),

  get: (taskId: string) =>
    http.get<any, any>(`/kb/comparison/${taskId}`),
}

// ── 审核中心 ──────────────────────────────────────────────────
export const auditApi = {
  list: (kbId: string, params?: { item_type?: string; status?: string }) =>
    http.get<any, any>('/kb/audit', { params: { kb_id: kbId, ...params } }),

  review: (itemId: string, kbId: string, action: 'approved' | 'rejected', comment?: string) =>
    http.put<any, any>(`/kb/audit/${itemId}`, null, {
      params: { kb_id: kbId, action, comment },
    }),
}
