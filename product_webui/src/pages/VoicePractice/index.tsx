import { useEffect, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Alert, Button, Card, Empty, Input, List, Modal, Select, Space, Spin, Tag, Typography, message,
} from 'antd'
import {
  AudioOutlined, CheckOutlined, CustomerServiceOutlined, LoadingOutlined,
  PauseOutlined, PlayCircleOutlined, ReloadOutlined, StopOutlined,
} from '@ant-design/icons'
import { voiceApi } from '../../api'
import { SourceReader } from '../../components/SourceDrawer'
import type { Citation, VoicePracticeSession, VoicePracticeTurn } from '../../types'

const { Title, Text, Paragraph } = Typography

const GOALS = [
  { value: 'free', label: '自由问答', description: '围绕资料自由交流' },
  { value: 'recall', label: '资料复述', description: '练习用自己的话讲清资料' },
  { value: 'interview', label: '面试练习', description: '模拟基于资料的追问' },
  { value: 'review', label: '重点回顾', description: '复习近期需要掌握的内容' },
]

type RecorderState = 'idle' | 'recording' | 'transcribing' | 'ready' | 'answering'

export default function VoicePracticePage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [goal, setGoal] = useState('free')
  const [session, setSession] = useState<VoicePracticeSession | null>(null)
  const [sessions, setSessions] = useState<VoicePracticeSession[]>([])
  const [turns, setTurns] = useState<VoicePracticeTurn[]>([])
  const [transcript, setTranscript] = useState('')
  const [recorderState, setRecorderState] = useState<RecorderState>('idle')
  const [error, setError] = useState('')
  const [micDenied, setMicDenied] = useState(false)
  const [source, setSource] = useState<{ citation: Citation; index: number } | null>(null)
  const [playing, setPlaying] = useState('')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const audioUrlRef = useRef('')
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const loadSessions = () => voiceApi.listSessions(kbId).then(value => setSessions(value.items || [])).catch(() => setSessions([]))

  useEffect(() => { void loadSessions() }, [kbId])
  useEffect(() => () => {
    recorderRef.current?.stop()
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
  }, [])

  const begin = async () => {
    setError('')
    try {
      const created = await voiceApi.createSession({ kb_id: kbId, goal })
      setSession(created)
      setTurns([])
      setTranscript('')
      setRecorderState('idle')
      await loadSessions()
    } catch (err) {
      setError(typeof err === 'string' ? err : '无法开始语音练习')
    }
  }

  const transcribe = async (blob: Blob) => {
    setRecorderState('transcribing')
    setError('')
    try {
      const form = new FormData()
      form.append('file', blob, 'recording.webm')
      const result = await voiceApi.transcribe(form)
      setTranscript(result.text || '')
      setRecorderState('ready')
      if (!result.text) setError('没有识别到可用语音内容')
    } catch (err) {
      setRecorderState('idle')
      setError(typeof err === 'string' ? err : '语音转文字失败，请重新录音')
    }
  }

  const startRecording = async () => {
    setError('')
    if (!session) {
      await begin()
      return
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('当前浏览器不支持录音，请使用最新版 Chrome 或 Edge')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = event => { if (event.data.size) chunksRef.current.push(event.data) }
      recorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop())
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        if (blob.size) void transcribe(blob)
        else setRecorderState('idle')
      }
      recorderRef.current = recorder
      recorder.start()
      setRecorderState('recording')
    } catch {
      setMicDenied(true)
      setError('麦克风权限被拒绝。请在浏览器地址栏允许麦克风后重试。')
    }
  }

  const stopRecording = () => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop()
      setRecorderState('transcribing')
    }
  }

  const submitTurn = async () => {
    if (!session || !transcript.trim() || recorderState === 'answering') return
    setRecorderState('answering')
    setError('')
    try {
      const turn = await voiceApi.turn(session.id, { kb_id: kbId, transcript: transcript.trim() })
      setTurns(items => [...items, { ...turn, answer: turn.answer || '', citations: turn.citations || [], memory_refs: turn.memories || [] }])
      setTranscript('')
      setRecorderState('idle')
    } catch (err) {
      setRecorderState('ready')
      setError(typeof err === 'string' ? err : '本轮回答失败，请重试')
    }
  }

  const playAnswer = async (turn: VoicePracticeTurn) => {
    if (!turn.answer) return
    setPlaying(turn.id)
    try {
      const blob = await voiceApi.speech(turn.id, kbId)
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
      const url = URL.createObjectURL(blob)
      audioUrlRef.current = url
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onended = () => setPlaying('')
      await audio.play()
    } catch (err) {
      setPlaying('')
      message.error(typeof err === 'string' ? err : '语音合成失败，请重试')
    }
  }

  const finish = async () => {
    if (!session) return
    try {
      const done = await voiceApi.finish(session.id, { kb_id: kbId })
      setSession(done)
      await loadSessions()
      message.success('练习已保存')
    } catch (err) {
      setError(typeof err === 'string' ? err : '结束练习失败，请重试')
    }
  }

  const loadHistory = async (item: VoicePracticeSession) => {
    try {
      const full = await voiceApi.getSession(item.id, kbId)
      setSession(full)
      setTurns(full.turns || [])
      setGoal(full.goal || 'free')
      setTranscript('')
      setRecorderState('idle')
      setError('')
    } catch {
      message.error('练习记录不可用')
    }
  }

  const currentGoal = GOALS.find(item => item.value === goal)

  return (
    <div data-testid="voice-practice-page" style={{ padding: 16, maxWidth: 1040, margin: '0 auto' }}>
      <Space direction="vertical" size={16} style={{ width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div>
            <Title level={4} style={{ margin: 0 }}>语音练习</Title>
            <Text type="secondary">用资料练习表达，回答仍遵循可信来源和个人记忆边界。</Text>
          </div>
          {session?.status === 'active' ? <Button icon={<CheckOutlined />} onClick={() => void finish()} data-testid="voice-finish">结束练习</Button> : null}
        </div>

        {error ? <Alert type="error" showIcon message={error} closable onClose={() => setError('')} /> : null}
        {micDenied ? <Alert type="warning" showIcon message="需要麦克风权限才能录音。你也可以直接编辑转写文本进行练习。" /> : null}

        {!session || session.status !== 'active' ? (
          <Card title="选择练习目标" data-testid="voice-start-card">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Select value={goal} onChange={setGoal} options={GOALS.map(item => ({ value: item.value, label: `${item.label} · ${item.description}` }))} style={{ maxWidth: 420, width: '100%' }} aria-label="练习目标" />
              <Button type="primary" icon={<AudioOutlined />} onClick={() => void begin()} data-testid="voice-start">开始练习</Button>
            </Space>
          </Card>
        ) : null}

        {session?.status === 'active' ? (
          <Card title={currentGoal?.label || '练习进行中'} data-testid="voice-session">
            <Space direction="vertical" size={14} style={{ width: '100%' }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <Button
                  type={recorderState === 'recording' ? 'default' : 'primary'}
                  danger={recorderState === 'recording'}
                  icon={recorderState === 'recording' ? <StopOutlined /> : <AudioOutlined />}
                  onClick={recorderState === 'recording' ? stopRecording : () => void startRecording()}
                  disabled={recorderState === 'transcribing' || recorderState === 'answering'}
                  data-testid={recorderState === 'recording' ? 'voice-stop' : 'voice-record'}
                >{recorderState === 'recording' ? '停止录音' : '开始录音'}</Button>
                {recorderState === 'transcribing' ? <Tag icon={<LoadingOutlined />} color="processing">正在转写</Tag> : null}
                {recorderState === 'answering' ? <Tag icon={<LoadingOutlined />} color="processing">正在回答</Tag> : null}
              </div>
              <Input.TextArea
                value={transcript}
                onChange={event => { setTranscript(event.target.value); if (recorderState === 'idle') setRecorderState('ready') }}
                placeholder="录音转写会出现在这里，也可以直接编辑后提交"
                rows={4}
                disabled={recorderState === 'transcribing' || recorderState === 'answering'}
                data-testid="voice-transcript"
              />
              <Space wrap>
                <Button icon={<ReloadOutlined />} onClick={() => { setTranscript(''); setRecorderState('idle') }} disabled={!transcript || recorderState === 'answering'}>重新录音</Button>
                <Button type="primary" icon={<PlayCircleOutlined />} onClick={() => void submitTurn()} disabled={!transcript.trim() || recorderState === 'answering'} data-testid="voice-submit">提交本轮</Button>
              </Space>
            </Space>
          </Card>
        ) : null}

        {session && turns.length ? (
          <Card title="本次练习" data-testid="voice-turns">
            <List
              dataSource={turns}
              renderItem={(turn, index) => (
                <List.Item style={{ display: 'block' }}>
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    <Text strong>第 {index + 1} 轮</Text>
                    <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}><Text type="secondary">你：</Text>{turn.transcript}</Paragraph>
                    {turn.answer ? <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}><Text type="secondary">回答：</Text>{turn.answer}</Paragraph> : <Text type="danger">本轮回答失败</Text>}
                    <Space wrap>
                      <Button size="small" icon={<CustomerServiceOutlined />} loading={playing === turn.id} onClick={() => void playAnswer(turn)} disabled={!turn.answer} data-testid="voice-play">播放回答</Button>
                      {(turn.citations || []).map((citation, citeIndex) => (
                        <Button key={citation.citation_id || citeIndex} size="small" onClick={() => setSource({ citation, index: citeIndex })}>[C{citeIndex + 1}] {citation.doc_name || '来源'}</Button>
                      ))}
                      {(turn.memory_refs || []).map((memory, memoryIndex) => <Tag key={memory.memory_id || memoryIndex} color="purple">[M{memoryIndex + 1}] {memory.content || '个人记忆'}</Tag>)}
                    </Space>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        ) : null}

        <Card title="历史练习" extra={<Button size="small" icon={<ReloadOutlined />} onClick={() => void loadSessions()}>刷新</Button>}>
          {sessions.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="还没有练习记录" /> : (
            <List
              dataSource={sessions}
              renderItem={item => <List.Item actions={[<Button key="open" type="link" onClick={() => void loadHistory(item)}>打开</Button>]}>
                <List.Item.Meta title={GOALS.find(goalItem => goalItem.value === item.goal)?.label || '语音练习'} description={`${item.status === 'active' ? '进行中' : '已完成'} · ${item.updated_at ? new Date(item.updated_at).toLocaleString() : ''}`} />
              </List.Item>}
            />
          )}
        </Card>
      </Space>
      <SourceReader open={Boolean(source)} citation={source?.citation || null} index={source?.index || 0} kbId={kbId} onClose={() => setSource(null)} />
    </div>
  )
}
