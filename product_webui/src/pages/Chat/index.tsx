import { useEffect, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Typography, Input, Button, Space, Tag, Spin, Avatar, Tooltip, Drawer, Switch, Radio, message, Dropdown, Modal,
} from 'antd'
import {
  SendOutlined, UserOutlined, RobotOutlined, ClearOutlined,
  CopyOutlined, ReloadOutlined, StopOutlined, SettingOutlined,
  PlusOutlined, MenuOutlined, PushpinOutlined, DeleteOutlined, InboxOutlined, EditOutlined,
  LikeOutlined, DislikeOutlined, BookOutlined,
} from '@ant-design/icons'
import { authHeader, forceRefresh, kbApi, memoryApi, workspaceApi } from '../../api'
import { CHAT_STARTERS } from '../../constants'
import { SourceDrawer } from '../../components/SourceDrawer'
import type { ChatEvent, Citation, ConversationItem, KbSettings, StoredMessage } from '../../types'

const { Title, Text, Paragraph } = Typography

interface Message {
  id?: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  loading?: boolean
  stopped?: boolean
  error?: string
  degraded?: boolean
  empty?: boolean
  question?: string
  status?: string
  memories?: { marker?: string; content?: string }[]
}

function fromStored(items: StoredMessage[]): Message[] {
  const result: Message[] = []
  let lastUser = ''
  for (const item of items) {
    if (item.role === 'user') {
      lastUser = item.content
      result.push({ id: item.id, role: 'user', content: item.content, status: item.status })
      continue
    }
    result.push({
      id: item.id,
      role: 'assistant',
      content: item.status === 'completed' ? item.content : '',
      citations: item.citations,
      stopped: item.status === 'stopped',
      error: item.status === 'failed' ? '问答服务暂时不可用' : undefined,
      empty: item.status === 'completed' && !item.content,
      question: lastUser,
      status: item.status,
      memories: item.retrieval_meta?.memories || [],
    })
  }
  return result
}

export default function ChatPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.innerWidth <= 768)
  const mobile = narrow
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [settings, setSettings] = useState<KbSettings | null>(null)
  const [sending, setSending] = useState(false)
  const [prefsOpen, setPrefsOpen] = useState(false)
  const [source, setSource] = useState<{ citation: Citation; index: number } | null>(null)
  const [conversations, setConversations] = useState<ConversationItem[]>([])
  const [conversationId, setConversationId] = useState('')
  const [search, setSearch] = useState('')
  const [listOpen, setListOpen] = useState(false)
  const [archived, setArchived] = useState(false)
  const [dislike, setDislike] = useState<Message | null>(null)
  const [reason, setReason] = useState('answer_wrong')
  const [comment, setComment] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const settingsRef = useRef<KbSettings | null>(null)
  const conversationRef = useRef('')

  useEffect(() => {
    conversationRef.current = conversationId
  }, [conversationId])

  const loadConversations = async (selected?: string) => {
    const page = await workspaceApi.listConversations(kbId, { q: search, archived })
    setConversations(page.items || [])
    const next = selected || conversationRef.current
    if (next && (page.items || []).some(item => item.id === next)) return
    if (next) {
      setConversationId('')
      conversationRef.current = ''
    }
  }

  useEffect(() => {
    const onResize = () => setNarrow(window.innerWidth <= 768)
    window.addEventListener('resize', onResize)
    onResize()
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    kbApi.getSettings(kbId).then((value) => {
      setSettings(value)
      settingsRef.current = value
    }).catch(() => {})
  }, [kbId])

  useEffect(() => {
    loadConversations().catch(() => setConversations([]))
  }, [kbId, search, archived])

  useEffect(() => {
    if (!conversationId) {
      setMessages([])
      return
    }
    workspaceApi.listMessages(conversationId, kbId).then((page) => {
      setMessages(fromStored(page.items || []))
    }).catch(() => setMessages([]))
  }, [conversationId, kbId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const savePrefs = async (patch: Partial<KbSettings>) => {
    const next = { ...(settingsRef.current || {}), ...patch }
    const saved = await kbApi.updateSettings(kbId, next)
    setSettings(saved)
    settingsRef.current = saved
    message.success('回答偏好已保存')
  }

  const startNew = async () => {
    abortRef.current?.abort()
    const created = await workspaceApi.createConversation(kbId, '新会话')
    setConversationId(created.id)
    setMessages([])
    setArchived(false)
    await loadConversations(created.id)
    setListOpen(false)
  }

  const ask = async (question: string, mode: 'new' | 'retry' = 'new') => {
    if (!question.trim() || sending) return
    const current = settingsRef.current
    const prior = mode === 'retry' ? messages.slice(0, -1) : messages
    setMessages(() => [
      ...(mode === 'new' ? [...prior, { role: 'user' as const, content: question }] : prior),
      { role: 'assistant', content: '', loading: true, question },
    ])
    setSending(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      let activeId = conversationRef.current
      if (!activeId) {
        const created = await workspaceApi.createConversation(kbId, question.slice(0, 32))
        activeId = created.id
        setConversationId(created.id)
        conversationRef.current = created.id
      }
      const payload = JSON.stringify({
        query: question,
        kb_id: kbId,
        conversation_id: activeId,
        retrieval_mode: current?.retrieval_mode,
        graph_enabled: current?.graph_enabled,
        top_k: current?.top_k,
        similarity_ratio: current?.similarity_ratio,
        enable_rerank: current?.rerank_enabled,
        similarity_threshold: current?.similarity_threshold,
        context_expand: current?.context_expand,
        answer_detail: current?.answer_detail === 'brief' ? 'concise' : current?.answer_detail === 'normal' || !current?.answer_detail ? 'standard' : current.answer_detail,
      })
      let response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader() },
        body: payload,
        signal: controller.signal,
      })
      if (response.status === 401 && await forceRefresh()) {
        response = await fetch('/api/v1/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeader() },
          body: payload,
          signal: controller.signal,
        })
      }
      if (!response.ok || !response.body) {
        throw new Error('问答服务暂时不可用')
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let answer = ''
      let citations: Citation[] = []
      let degraded = false
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() || ''
        for (const frame of frames) {
          const line = frame.split('\n').find(item => item.startsWith('data: '))
          if (!line) continue
          const event = JSON.parse(line.slice(6)) as ChatEvent
          if (event.conversation_id) {
            setConversationId(event.conversation_id)
            conversationRef.current = event.conversation_id
          }
          if (event.type === 'meta') {
            citations = event.citations || []
            degraded = Boolean(event.degraded)
          }
          if (event.type === 'token') answer += event.text || ''
          if (event.type === 'done') {
            if (event.answer) answer = event.answer
            if (event.citations) citations = event.citations
            if (event.answerable === false) citations = []
          }
          if (event.type === 'token' || event.type === 'done') {
            const snapshot = answer
            const cites = citations
            setMessages(prev => [
              ...prev.slice(0, -1),
              { role: 'assistant', content: snapshot, citations: cites, question, degraded },
            ])
          }
          if (event.type === 'error') throw new Error(event.message || '问答服务暂时不可用')
        }
      }
      if (!answer) {
        setMessages(prev => [
          ...prev.slice(0, -1),
          { role: 'assistant', content: '', citations, empty: true, question, degraded },
        ])
      }
      await loadConversations(conversationRef.current)
      if (conversationRef.current) {
        const stored = await workspaceApi.listMessages(conversationRef.current, kbId)
        setMessages(fromStored(stored.items || []))
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setMessages(prev => {
          const last = prev[prev.length - 1]
          return [...prev.slice(0, -1), { ...last, loading: false, stopped: true }]
        })
      } else {
        const text = err instanceof Error ? err.message : '问答服务暂时不可用'
        setMessages(prev => [
          ...prev.slice(0, -1),
          { role: 'assistant', content: '', error: text, question, degraded: true },
        ])
      }
    } finally {
      abortRef.current = null
      setSending(false)
    }
  }

  const send = () => {
    const question = input.trim()
    if (!question) return
    setInput('')
    void ask(question)
  }

  const showCitations = settings?.show_citations !== false
  const current = conversations.find(item => item.id === conversationId)

  const list = (
    <div data-testid="conversation-list" style={{ display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0 }}>
      <Space style={{ padding: 12, width: '100%' }}>
        <Button data-testid="conversation-new" type="primary" icon={<PlusOutlined />} onClick={() => void startNew()}>新建</Button>
      </Space>
      <Input
        data-testid="conversation-search"
        placeholder="搜索会话"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        style={{ margin: '0 12px 8px' }}
      />
      <Button type="link" onClick={() => setArchived(value => !value)}>
        {archived ? '查看进行中' : '查看归档'}
      </Button>
      <div style={{ flex: 1, overflow: 'auto', padding: '0 8px 8px' }}>
        {conversations.map(item => (
          <div
            key={item.id}
            data-testid="conversation-item"
            onClick={() => {
              setConversationId(item.id)
              setListOpen(false)
            }}
            style={{
              padding: 10,
              borderRadius: 8,
              cursor: 'pointer',
              background: item.id === conversationId ? '#f0f0f0' : 'transparent',
              marginBottom: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <Text strong ellipsis style={{ flex: 1 }}>{item.is_pinned ? '📌 ' : ''}{item.title || '新会话'}</Text>
              <Dropdown
                trigger={['click']}
                menu={{
                  items: [
                    { key: 'rename', icon: <EditOutlined />, label: '重命名', onClick: () => {
                      const title = window.prompt('会话名称', item.title || '')
                      if (title != null) workspaceApi.patchConversation(item.id, { kb_id: kbId, title }).then(() => loadConversations(item.id))
                    } },
                    { key: 'pin', icon: <PushpinOutlined />, label: item.is_pinned ? '取消置顶' : '置顶', onClick: () => workspaceApi.pinConversation(item.id, kbId).then(() => loadConversations(item.id)) },
                    { key: 'archive', icon: <InboxOutlined />, label: item.is_archived ? '取消归档' : '归档', onClick: () => {
                      const action = item.is_archived
                        ? workspaceApi.patchConversation(item.id, { kb_id: kbId, is_archived: false })
                        : workspaceApi.archiveConversation(item.id, kbId)
                      action.then(() => loadConversations())
                    } },
                    { key: 'delete', icon: <DeleteOutlined />, label: '删除', danger: true, onClick: () => workspaceApi.deleteConversation(item.id, kbId).then(() => {
                      if (conversationId === item.id) {
                        setConversationId('')
                        setMessages([])
                      }
                      loadConversations()
                    }) },
                  ],
                }}
              >
                <Button size="small" type="text" onClick={(event) => event.stopPropagation()}>···</Button>
              </Dropdown>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  const chat = (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 16, minWidth: 0, flex: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 8 }}>
        <Space>
          {mobile ? <Button data-testid="conversation-open" icon={<MenuOutlined />} onClick={() => setListOpen(true)} /> : null}
          <Title level={4} style={{ margin: 0 }}>{current?.title || '问答'}</Title>
        </Space>
        <Space>
          <Tooltip title="回答偏好">
            <Button size="small" aria-label="回答偏好" icon={<SettingOutlined />} onClick={() => setPrefsOpen(true)} />
          </Tooltip>
          <Tooltip title="清空当前会话">
            <Button
              size="small"
              aria-label="清空当前会话"
              data-testid="chat-clear"
              icon={<ClearOutlined />}
              onClick={() => void startNew()}
            />
          </Tooltip>
        </Space>
      </div>

      <div style={{
        flex: 1, overflow: 'auto', marginBottom: 12, minWidth: 0,
        background: '#fff', borderRadius: 8, border: '1px solid #f0f0f0', padding: 16,
      }}>
        {messages.length === 0 ? (
          <div style={{ minHeight: 240, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Space direction="vertical" align="center">
              <RobotOutlined style={{ fontSize: 28, color: '#595959' }} />
              <Text type="secondary">根据个人资料回答，并标出来源</Text>
              <Space wrap style={{ marginTop: 8, justifyContent: 'center' }}>
                {CHAT_STARTERS.map(item => (
                  <Tag key={item} style={{ cursor: 'pointer', padding: '4px 12px', whiteSpace: 'normal' }} onClick={() => setInput(item)}>
                    {item}
                  </Tag>
                ))}
              </Space>
            </Space>
          </div>
        ) : (
          <Space direction="vertical" style={{ width: '100%' }} size={16}>
            {messages.map((msg, i) => (
              <div key={msg.id || i} style={{ display: 'flex', gap: 12, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', minWidth: 0 }}>
                <Avatar
                  icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                  style={{ background: msg.role === 'user' ? '#595959' : '#f5f5f5', color: msg.role === 'user' ? '#fff' : '#595959', flexShrink: 0 }}
                />
                <div style={{ maxWidth: '100%', minWidth: 0, flex: 1 }}>
                  <div style={{
                    background: msg.role === 'user' ? '#f0f0f0' : '#fafafa',
                    padding: '10px 14px', borderRadius: 8, fontSize: 14, lineHeight: 1.7,
                    overflowWrap: 'anywhere',
                  }}>
                    {msg.loading ? <Spin size="small" /> : null}
                    {msg.empty ? (
                      <div data-testid="chat-no-evidence">资料里没有可引用的内容。</div>
                    ) : null}
                    {msg.error ? (
                      <div data-testid="chat-error">
                        <Text type="danger">{msg.error}</Text>
                        {msg.degraded ? <div><Tag>服务降级</Tag></div> : null}
                      </div>
                    ) : null}
                    {msg.content ? (
                      <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                        {msg.content.split(/(\[C\d+\]|\[M\d+\])/g).map((part, partIndex) => {
                          const cite = part.match(/^\[(C\d+)\]$/)
                          if (cite) {
                            const cited = msg.citations?.find(item => item.citation_id === cite[1])
                            if (!cited) return <span key={partIndex}>{part}</span>
                            const position = (msg.citations || []).indexOf(cited)
                            return (
                              <Button
                                key={partIndex}
                                type="link"
                                size="small"
                                style={{ padding: 0, height: 'auto' }}
                                data-testid="citation-mark"
                                onClick={() => setSource({ citation: cited, index: position + 1 })}
                              >
                                {part}
                              </Button>
                            )
                          }
                          if (/^\[M\d+\]$/.test(part)) {
                            return <Tag key={partIndex} data-testid="memory-mark" color="purple">{part} 个人记忆</Tag>
                          }
                          return <span key={partIndex}>{part}</span>
                        })}
                      </Paragraph>
                    ) : null}
                    {msg.stopped ? <Text type="secondary">已停止</Text> : null}
                  </div>
                  {msg.role === 'assistant' && !msg.loading ? (
                    <Space wrap style={{ marginTop: 8 }}>
                      <Tooltip title="复制答案">
                        <Button
                          size="small"
                          aria-label="复制答案"
                          data-testid="chat-copy"
                          icon={<CopyOutlined />}
                          onClick={() => navigator.clipboard.writeText(msg.content || msg.error || '')}
                        />
                      </Tooltip>
                      {msg.question ? (
                        <Tooltip title="重新回答">
                          <Button
                            size="small"
                            aria-label="重新回答"
                            data-testid="chat-retry"
                            icon={<ReloadOutlined />}
                            onClick={() => ask(msg.question || '', 'retry')}
                          />
                        </Tooltip>
                      ) : null}
                      {msg.id ? (
                        <>
                          <Tooltip title="有用">
                            <Button size="small" data-testid="chat-like" icon={<LikeOutlined />} onClick={() => {
                              workspaceApi.feedback(msg.id || '', { kb_id: kbId, rating: 'positive' }).then(() => message.success('已记录为有用'))
                            }} />
                          </Tooltip>
                          <Tooltip title="无用">
                            <Button size="small" data-testid="chat-dislike" icon={<DislikeOutlined />} onClick={() => { setDislike(msg); setReason('answer_wrong'); setComment('') }} />
                          </Tooltip>
                          <Tooltip title="保存为记忆">
                            <Button size="small" data-testid="save-memory" icon={<BookOutlined />} onClick={() => {
                              const content = (msg.content || '').trim()
                              if (!content) return
                              memoryApi.createCandidate({
                                kb_id: kbId,
                                content: content.slice(0, 800),
                                conversation_id: conversationId,
                                message_id: msg.id,
                                category: 'other',
                              }).then(() => message.success('已加入待确认记忆'))
                            }} />
                          </Tooltip>
                        </>
                      ) : null}
                    </Space>
                  ) : null}
                  {showCitations && msg.citations && msg.citations.length > 0 ? (
                    <div style={{ marginTop: 8 }}>
                      <Text type="secondary">来自资料</Text>
                      <Space wrap>
                      {msg.citations.map((citation, index) => (
                        <Button
                          key={`${citation.chunk_id || index}`}
                          size="small"
                          data-testid="citation-open"
                          aria-label={`打开引用 ${index + 1}`}
                          onClick={() => setSource({ citation, index: index + 1 })}
                        >
                          [{citation.citation_id || index + 1}] {citation.doc_name || '资料'}
                        </Button>
                      ))}
                      </Space>
                    </div>
                  ) : null}
                  {(msg.memories || []).length > 0 ? (
                    <div style={{ marginTop: 8 }}>
                      <Text type="secondary">来自个人记忆</Text>
                      <Space wrap>
                        {(msg.memories || []).map((item, index) => (
                          <Tag key={item.marker || index} data-testid="memory-chip">{item.marker || `M${index + 1}`} {item.content}</Tag>
                        ))}
                      </Space>
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </Space>
        )}
      </div>

      <div style={{
        background: '#fff', borderRadius: 8, border: '1px solid #e8e8e8',
        padding: 8, display: 'flex', alignItems: 'flex-end', gap: 8, minWidth: 0,
      }}>
        <Input.TextArea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="输入问题，Shift+Enter 换行，Enter 发送"
          autoSize={{ minRows: 1, maxRows: 5 }}
          style={{ border: 'none', resize: 'none', padding: 4, fontSize: 14 }}
          onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); send() } }}
        />
        {sending ? (
          <Tooltip title="停止生成">
            <Button
              aria-label="停止生成"
              data-testid="chat-stop"
              icon={<StopOutlined />}
              onClick={() => abortRef.current?.abort()}
            />
          </Tooltip>
        ) : (
          <Tooltip title="发送">
            <Button
              type="primary"
              aria-label="发送"
              icon={<SendOutlined />}
              onClick={send}
              disabled={!input.trim()}
            />
          </Tooltip>
        )}
      </div>
    </div>
  )

  return (
    <div style={{ height: '100%', minHeight: 'calc(100vh - 112px)', display: 'flex', minWidth: 0 }}>
      {mobile ? null : (
        <div style={{ width: 260, flexShrink: 0, borderRight: '1px solid #f0f0f0', background: '#fff' }}>{list}</div>
      )}
      {chat}
      <Drawer
        title="会话"
        open={listOpen}
        onClose={() => setListOpen(false)}
        placement="left"
        width="100%"
        data-testid="conversation-drawer"
      >
        {list}
      </Drawer>
      <Drawer title="回答偏好" open={prefsOpen} onClose={() => setPrefsOpen(false)} width={Math.min(400, window.innerWidth - 16)}>
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <div>
            <Text>使用知识关联</Text>
            <Switch
              style={{ marginLeft: 12 }}
              checked={settings?.graph_enabled !== false}
              onChange={(checked) => savePrefs({ graph_enabled: checked })}
            />
          </div>
          <div>
            <Text>扩展上下文</Text>
            <Switch
              style={{ marginLeft: 12 }}
              checked={Boolean(settings?.context_expand)}
              onChange={(checked) => savePrefs({ context_expand: checked })}
            />
          </div>
          <div>
            <Text>回答详细程度</Text>
            <Radio.Group
              style={{ display: 'block', marginTop: 8 }}
              value={settings?.answer_detail === 'brief' ? 'concise' : settings?.answer_detail === 'normal' || !settings?.answer_detail ? 'standard' : settings.answer_detail}
              onChange={(event) => savePrefs({ answer_detail: event.target.value })}
            >
              <Radio.Button value="concise">简洁</Radio.Button>
              <Radio.Button value="standard">标准</Radio.Button>
              <Radio.Button value="detailed">详细</Radio.Button>
            </Radio.Group>
          </div>
          <div>
            <Text>默认显示引用</Text>
            <Switch
              style={{ marginLeft: 12 }}
              checked={settings?.show_citations !== false}
              onChange={(checked) => savePrefs({ show_citations: checked })}
            />
          </div>
        </Space>
      </Drawer>
      <Modal
        title="反馈原因"
        open={Boolean(dislike)}
        onCancel={() => setDislike(null)}
        onOk={() => {
          if (!dislike?.id) return
          workspaceApi.feedback(dislike.id, { kb_id: kbId, rating: 'negative', reason, comment }).then(() => {
            message.success('已记录反馈')
            setDislike(null)
          })
        }}
      >
        <Radio.Group value={reason} onChange={(event) => setReason(event.target.value)}>
          <Space direction="vertical">
            <Radio value="answer_wrong">回答错误</Radio>
            <Radio value="citation_wrong">引用错误</Radio>
            <Radio value="not_found">资料不足</Radio>
            <Radio value="outdated">内容过时</Radio>
            <Radio value="unclear">表达不清</Radio>
            <Radio value="other">其他</Radio>
          </Space>
        </Radio.Group>
        <Input.TextArea style={{ marginTop: 12 }} value={comment} onChange={(event) => setComment(event.target.value)} placeholder="补充说明（可选）" />
      </Modal>
      <SourceDrawer
        open={Boolean(source)}
        citation={source?.citation || null}
        index={source?.index || 1}
        kbId={kbId}
        onClose={() => setSource(null)}
      />
    </div>
  )
}
