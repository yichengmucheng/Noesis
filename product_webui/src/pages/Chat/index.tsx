import { useEffect, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Typography, Input, Button, Space, Tag, Spin, Avatar, Tooltip, Drawer, Switch, Radio, message,
} from 'antd'
import {
  SendOutlined, UserOutlined, RobotOutlined, ClearOutlined,
  CopyOutlined, ReloadOutlined, StopOutlined, SettingOutlined,
} from '@ant-design/icons'
import { authHeader, forceRefresh, kbApi } from '../../api'
import { CHAT_STARTERS } from '../../constants'
import { SourceDrawer } from '../../components/SourceDrawer'
import type { ChatEvent, Citation, KbSettings } from '../../types'

const { Title, Text, Paragraph } = Typography

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  loading?: boolean
  stopped?: boolean
  error?: string
  degraded?: boolean
  empty?: boolean
  question?: string
}

export default function ChatPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [settings, setSettings] = useState<KbSettings | null>(null)
  const [sending, setSending] = useState(false)
  const [prefsOpen, setPrefsOpen] = useState(false)
  const [source, setSource] = useState<{ citation: Citation; index: number } | null>(null)
  const [sessionId, setSessionId] = useState(() => localStorage.getItem(`kb-session:${kbId}`) || crypto.randomUUID())
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const settingsRef = useRef<KbSettings | null>(null)

  useEffect(() => {
    localStorage.setItem(`kb-session:${kbId}`, sessionId)
  }, [kbId, sessionId])

  useEffect(() => {
    kbApi.getSettings(kbId).then((value) => {
      setSettings(value)
      settingsRef.current = value
    }).catch(() => {})
  }, [kbId])

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

  const ask = async (question: string, mode: 'new' | 'retry' = 'new') => {
    if (!question.trim() || sending) return
    const current = settingsRef.current
    const prior = mode === 'retry' ? messages.slice(0, -1) : messages
    const history = prior
      .filter(item => item.content && !item.loading && !item.error)
      .map(item => ({ role: item.role, content: item.content }))
    setMessages(() => [
      ...(mode === 'new' ? [...prior, { role: 'user' as const, content: question }] : prior),
      { role: 'assistant', content: '', loading: true, question },
    ])
    setSending(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const payload = JSON.stringify({
        query: question,
        kb_id: kbId,
        retrieval_mode: current?.retrieval_mode,
        graph_enabled: current?.graph_enabled,
        top_k: current?.top_k,
        similarity_ratio: current?.similarity_ratio,
        enable_rerank: current?.rerank_enabled,
        similarity_threshold: current?.similarity_threshold,
        context_expand: current?.context_expand,
        history,
        session_id: sessionId,
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
          if (event.type === 'meta') {
            citations = event.citations || []
            degraded = Boolean(event.degraded)
          }
          if (event.type === 'token') {
            answer += event.text || ''
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

  return (
    <div style={{ height: '100%', minHeight: 'calc(100vh - 112px)', display: 'flex', flexDirection: 'column', padding: 16, minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 8 }}>
        <Title level={4} style={{ margin: 0 }}>问答</Title>
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
              onClick={() => {
                abortRef.current?.abort()
                setMessages([])
                setSessionId(crypto.randomUUID())
              }}
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
              <div key={i} style={{ display: 'flex', gap: 12, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row', minWidth: 0 }}>
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
                      <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.content}</Paragraph>
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
                    </Space>
                  ) : null}
                  {showCitations && msg.citations && msg.citations.length > 0 ? (
                    <Space wrap style={{ marginTop: 8 }}>
                      {msg.citations.map((citation, index) => (
                        <Button
                          key={`${citation.chunk_id || index}`}
                          size="small"
                          data-testid="citation-open"
                          aria-label={`打开引用 ${index + 1}`}
                          onClick={() => setSource({ citation, index: index + 1 })}
                        >
                          [{index + 1}] {citation.doc_name || '资料'}
                        </Button>
                      ))}
                    </Space>
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
              value={settings?.answer_detail || 'normal'}
              onChange={(event) => savePrefs({ answer_detail: event.target.value })}
            >
              <Radio.Button value="brief">简要</Radio.Button>
              <Radio.Button value="normal">标准</Radio.Button>
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
      <SourceDrawer
        open={Boolean(source)}
        citation={source?.citation || null}
        index={source?.index || 1}
        onClose={() => setSource(null)}
      />
    </div>
  )
}
