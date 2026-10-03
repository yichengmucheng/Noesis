import { useEffect, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Alert, Button, Card, Col, Drawer, Empty, Grid, Input, List, Modal, Row, Select, Space,
  Spin, Statistic, Tag, Typography, message,
} from 'antd'
import {
  AudioOutlined, CheckOutlined, CustomerServiceOutlined, LoadingOutlined,
  PauseOutlined, PlayCircleOutlined, ReloadOutlined, RightOutlined, StopOutlined,
} from '@ant-design/icons'
import { realtimeVoiceSocketUrl, voiceApi } from '../../api'
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
type RealtimeState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'paused' | 'stopped'

export default function VoicePracticePage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const screens = Grid.useBreakpoint()
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
  const realtimeSocketRef = useRef<WebSocket | null>(null)
  const realtimeAudioContextRef = useRef<AudioContext | null>(null)
  const realtimeSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const realtimeProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const realtimeStreamRef = useRef<MediaStream | null>(null)
  const realtimeSpeechRef = useRef(false)
  const realtimeSilenceRef = useRef(0)
  const realtimeTurnSentRef = useRef(false)
  const realtimeTranscriptRef = useRef('')
  const realtimeAudioElementRef = useRef<HTMLAudioElement | null>(null)
  const realtimeMediaSourceRef = useRef<MediaSource | null>(null)
  const realtimeSourceBufferRef = useRef<SourceBuffer | null>(null)
  const realtimeAudioQueueRef = useRef<ArrayBuffer[]>([])
  const realtimeSessionRef = useRef<VoicePracticeSession | null>(null)
  const realtimeReconnectTimerRef = useRef<number | null>(null)
  const realtimeReconnectAttemptRef = useRef(0)
  const realtimeManualStopRef = useRef(false)
  const realtimeAsrReadyRef = useRef(false)
  const [realtimeState, setRealtimeState] = useState<RealtimeState>('idle')
  const [realtimeTranscript, setRealtimeTranscript] = useState('')
  const [realtimeAnswer, setRealtimeAnswer] = useState('')
  const [realtimeCitations, setRealtimeCitations] = useState<Citation[]>([])
  const [realtimeError, setRealtimeError] = useState('')
  const [realtimeStatus, setRealtimeStatus] = useState('')
  const [historySession, setHistorySession] = useState<VoicePracticeSession | null>(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)

  const loadSessions = () => voiceApi.listSessions(kbId).then(value => setSessions(value.items || [])).catch(() => setSessions([]))

  useEffect(() => { void loadSessions() }, [kbId])
  useEffect(() => () => {
    recorderRef.current?.stop()
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
    realtimeSocketRef.current?.close()
    if (realtimeReconnectTimerRef.current !== null) window.clearTimeout(realtimeReconnectTimerRef.current)
    realtimeProcessorRef.current?.disconnect()
    realtimeSourceRef.current?.disconnect()
    realtimeStreamRef.current?.getTracks().forEach(track => track.stop())
    void realtimeAudioContextRef.current?.close()
  }, [])

  const releaseRealtimeTransport = () => {
    realtimeAsrReadyRef.current = false
    realtimeSocketRef.current?.close()
    realtimeSocketRef.current = null
    realtimeProcessorRef.current?.disconnect()
    realtimeSourceRef.current?.disconnect()
    realtimeStreamRef.current?.getTracks().forEach(track => track.stop())
    realtimeProcessorRef.current = null
    realtimeSourceRef.current = null
    realtimeStreamRef.current = null
    void realtimeAudioContextRef.current?.close()
    realtimeAudioContextRef.current = null
  }

  const downsamplePcm = (input: Float32Array, inputRate: number, outputRate = 16000) => {
    if (inputRate === outputRate) {
      const output = new Int16Array(input.length)
      for (let index = 0; index < input.length; index += 1) output[index] = Math.max(-1, Math.min(1, input[index])) * 0x7fff
      return output.buffer
    }
    const ratio = inputRate / outputRate
    const length = Math.max(1, Math.round(input.length / ratio))
    const output = new Int16Array(length)
    for (let index = 0; index < length; index += 1) {
      const start = Math.floor(index * ratio)
      const end = Math.min(input.length, Math.floor((index + 1) * ratio))
      let total = 0
      let count = 0
      for (let cursor = start; cursor < end; cursor += 1) { total += input[cursor]; count += 1 }
      output[index] = Math.max(-1, Math.min(1, count ? total / count : 0)) * 0x7fff
    }
    return output.buffer
  }

  const resetRealtimeAudio = () => {
    realtimeAudioElementRef.current?.pause()
    realtimeAudioElementRef.current = null
    realtimeMediaSourceRef.current = null
    realtimeSourceBufferRef.current = null
    realtimeAudioQueueRef.current = []
  }

  const flushRealtimeAudio = () => {
    const sourceBuffer = realtimeSourceBufferRef.current
    const queue = realtimeAudioQueueRef.current
    if (!sourceBuffer || sourceBuffer.updating || !queue.length) return
    try { sourceBuffer.appendBuffer(queue.shift() as ArrayBuffer) } catch { queue.length = 0 }
  }

  const startRealtimeAudio = () => {
    if (!('MediaSource' in window)) return
    resetRealtimeAudio()
    const mediaSource = new MediaSource()
    const audio = new Audio()
    audio.autoplay = true
    audio.controls = false
    audio.src = URL.createObjectURL(mediaSource)
    mediaSource.addEventListener('sourceopen', () => {
      try {
        const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg')
        sourceBuffer.addEventListener('updateend', flushRealtimeAudio)
        realtimeSourceBufferRef.current = sourceBuffer
        flushRealtimeAudio()
      } catch {
        setRealtimeError('当前浏览器不支持流式音频播放')
      }
    }, { once: true })
    realtimeMediaSourceRef.current = mediaSource
    realtimeAudioElementRef.current = audio
  }

  const cancelRealtime = (reason: 'barge_in' | 'cancel' = 'cancel') => {
    const socket = realtimeSocketRef.current
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: reason }))
    resetRealtimeAudio()
    setRealtimeState('listening')
    setRealtimeStatus('已打断，正在听')
    setRealtimeAnswer('')
    realtimeTurnSentRef.current = false
  }

  const openRealtime = async (activeSession: VoicePracticeSession) => {
    setRealtimeError('')
    setRealtimeState('connecting')
    setRealtimeStatus('正在连接语音服务')
    releaseRealtimeTransport()
    realtimeAsrReadyRef.current = false
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
      const context = new AudioContext({ sampleRate: 16000 })
      await context.resume()
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(4096, 1, 1)
      const socketInfo = realtimeVoiceSocketUrl()
      const socket = new WebSocket(socketInfo.url, socketInfo.protocols)
      socket.binaryType = 'arraybuffer'
      realtimeStreamRef.current = stream
      realtimeAudioContextRef.current = context
      realtimeSourceRef.current = source
      realtimeProcessorRef.current = processor
      realtimeSocketRef.current = socket
      processor.onaudioprocess = event => {
        if (socket.readyState !== WebSocket.OPEN || !realtimeAsrReadyRef.current) return
        const samples = event.inputBuffer.getChannelData(0)
        let energy = 0
        for (let index = 0; index < samples.length; index += 1) energy += samples[index] * samples[index]
        const rms = Math.sqrt(energy / Math.max(1, samples.length))
        const active = realtimeStateRef.current === 'listening' || realtimeStateRef.current === 'thinking' || realtimeStateRef.current === 'speaking'
        if (!active) return
        if (rms > 0.018) {
          if (!realtimeSpeechRef.current) {
            if (realtimeStateRef.current === 'thinking' || realtimeStateRef.current === 'speaking') cancelRealtime('barge_in')
            else {
              setRealtimeAnswer('')
              setRealtimeCitations([])
            }
          }
          realtimeSpeechRef.current = true
          realtimeSilenceRef.current = 0
        } else if (realtimeSpeechRef.current) {
          realtimeSilenceRef.current += (samples.length / context.sampleRate) * 1000
          if (realtimeSilenceRef.current >= 850 && !realtimeTurnSentRef.current) {
            realtimeTurnSentRef.current = true
            realtimeAsrReadyRef.current = false
            socket.send(JSON.stringify({ type: 'end_turn', text: realtimeTranscriptRef.current }))
            setRealtimeState('thinking')
            setRealtimeStatus('正在理解你的问题')
            realtimeSpeechRef.current = false
            realtimeSilenceRef.current = 0
          }
        }
        socket.send(downsamplePcm(samples, context.sampleRate))
      }
      source.connect(processor)
      processor.connect(context.destination)
      socket.onopen = () => socket.send(JSON.stringify({ type: 'start', kb_id: kbId, session_id: activeSession.id, goal: activeSession.goal || goal }))
      socket.onclose = () => {
        realtimeAsrReadyRef.current = false
        if (realtimeManualStopRef.current || realtimeStateRef.current === 'stopped') {
          setRealtimeState('stopped')
          return
        }
        releaseRealtimeTransport()
        const attempt = realtimeReconnectAttemptRef.current + 1
        realtimeReconnectAttemptRef.current = attempt
        if (attempt > 5) {
          setRealtimeState('stopped')
          setRealtimeError('实时连接已断开，重试次数已用尽，请重新开始')
          return
        }
        const delay = Math.min(1000 * (2 ** (attempt - 1)), 8000)
        setRealtimeState('connecting')
        setRealtimeError(`连接已断开，正在第 ${attempt} 次重连…`)
        realtimeReconnectTimerRef.current = window.setTimeout(() => {
          realtimeReconnectTimerRef.current = null
          void openRealtime(activeSession)
        }, delay)
      }
      socket.onerror = () => setRealtimeError('实时语音连接失败，请重试')
      socket.onmessage = event => {
        if (typeof event.data !== 'string') {
          const buffer = event.data instanceof ArrayBuffer ? event.data : null
          if (buffer) { realtimeAudioQueueRef.current.push(buffer); flushRealtimeAudio() }
          return
        }
        const payload = JSON.parse(event.data) as { type?: string; text?: string; answer?: string; citations?: Citation[]; message?: string; code?: string; stage?: string; phase?: string; retryable?: boolean; turn?: VoicePracticeTurn; session?: VoicePracticeSession }
        if (payload.type === 'ready') {
          realtimeReconnectAttemptRef.current = 0
          setRealtimeError('')
        }
        if (payload.type === 'asr_ready') {
          realtimeAsrReadyRef.current = true
          realtimeReconnectAttemptRef.current = 0
          setRealtimeError('')
          if (['idle', 'connecting', 'listening'].includes(realtimeStateRef.current)) {
            setRealtimeState('listening')
            setRealtimeStatus('可以继续说')
          }
        }
        if (payload.type === 'answer_status') {
          setRealtimeState('thinking')
          setRealtimeStatus(payload.message || '正在处理')
        }
        if (payload.type === 'transcript' || payload.type === 'transcript_final') {
          const text = payload.text || ''
          realtimeTranscriptRef.current = text
          setRealtimeTranscript(text)
          if (text) setRealtimeError('')
        }
        if (payload.type === 'answer_token') { setRealtimeState('speaking'); setRealtimeStatus('正在语音回答'); setRealtimeAnswer(value => value + (payload.text || '')) }
        if (payload.type === 'answer_done') { setRealtimeAnswer(payload.answer || ''); setRealtimeCitations(payload.citations || []); setRealtimeState('speaking'); setRealtimeStatus('正在语音回答') }
        if (payload.type === 'tts_start') startRealtimeAudio()
        if (payload.type === 'error') {
          if (payload.stage === 'asr') realtimeAsrReadyRef.current = false
          const nonRetryableCodes = new Set([
            'auth',
            'aliyun_not_configured',
            'aliyun_permission_denied',
            'aliyun_access_key_invalid',
            'realtime_dependency_missing',
          ])
          if (payload.retryable === false || (payload.code && nonRetryableCodes.has(payload.code))) {
            realtimeManualStopRef.current = true
          }
          setRealtimeError(payload.message || '实时语音服务失败')
        }
        if (payload.type === 'answer_cancelled') { resetRealtimeAudio(); setRealtimeAnswer(''); setRealtimeState('listening'); setRealtimeStatus('已打断，正在听'); realtimeTurnSentRef.current = false }
        if (payload.type === 'turn_done') {
          if (payload.turn) {
            setTurns(items => items.some(item => item.id === payload.turn?.id) ? items : [...items, payload.turn as VoicePracticeTurn])
            void loadSessions()
          }
          setRealtimeState('listening')
          setRealtimeStatus('可以继续说')
          realtimeTurnSentRef.current = false
          realtimeTranscriptRef.current = ''
          setRealtimeTranscript('')
        }
        if (payload.type === 'paused') setRealtimeState('paused')
        if (payload.type === 'resumed') { setRealtimeState('listening'); setRealtimeStatus('可以继续说') }
        if (payload.type === 'finished') {
          realtimeManualStopRef.current = true
          setRealtimeState('stopped')
          setRealtimeStatus('本次对话已结束')
          if (payload.session) setSession(payload.session)
          releaseRealtimeTransport()
          void loadSessions()
        }
      }
    } catch (err) {
      setRealtimeState('stopped')
      setRealtimeError(err instanceof Error ? err.message : '无法打开麦克风')
    }
  }

  const realtimeStateRef = useRef<RealtimeState>('idle')
  useEffect(() => { realtimeStateRef.current = realtimeState }, [realtimeState])

  const startRealtime = async () => {
    let active = session
    if (!active) {
      try {
        active = await voiceApi.createSession({ kb_id: kbId, goal })
        setSession(active)
        setTurns([])
        await loadSessions()
      } catch (err) {
        setRealtimeError(typeof err === 'string' ? err : '无法开始实时练习')
        return
      }
    }
    realtimeManualStopRef.current = false
    realtimeReconnectAttemptRef.current = 0
    realtimeSessionRef.current = active
    await openRealtime(active)
  }

  const pauseRealtime = () => {
    if (realtimeSocketRef.current?.readyState === WebSocket.OPEN) {
      const command = realtimeState === 'paused' ? 'resume' : 'pause'
      if (command === 'pause') realtimeAsrReadyRef.current = false
      realtimeSocketRef.current.send(JSON.stringify({ type: command }))
    }
  }

  const stopRealtime = () => {
    realtimeManualStopRef.current = true
    realtimeAsrReadyRef.current = false
    if (realtimeReconnectTimerRef.current !== null) {
      window.clearTimeout(realtimeReconnectTimerRef.current)
      realtimeReconnectTimerRef.current = null
    }
    if (realtimeSocketRef.current?.readyState === WebSocket.OPEN) realtimeSocketRef.current.send(JSON.stringify({ type: 'finish' }))
    releaseRealtimeTransport()
    resetRealtimeAudio()
    setRealtimeState('stopped')
  }

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
    setHistoryOpen(true)
    setHistoryLoading(true)
    setHistorySession({ ...item, turns: [] })
    try {
      const full = await voiceApi.getSession(item.id, kbId)
      setHistorySession(full)
    } catch {
      setHistoryOpen(false)
      setHistorySession(null)
      message.error('练习记录不可用')
    } finally {
      setHistoryLoading(false)
    }
  }

  const currentGoal = GOALS.find(item => item.value === goal)
  const historyTurns = historySession?.turns || []
  const historyCompletedTurns = historyTurns.filter(item => item.status === 'completed' && item.answer).length
  const historyCitedTurns = historyTurns.filter(item => (item.citations || []).length > 0).length
  const historyMemoryTurns = historyTurns.filter(item => (item.memory_refs || []).length > 0).length
  const historyFailedTurns = historyTurns.filter(item => item.status !== 'completed' || !item.answer).length
  const historyDurationMinutes = (() => {
    if (!historySession?.created_at) return 0
    const start = new Date(historySession.created_at).getTime()
    const end = new Date(historySession.ended_at || historySession.updated_at || '').getTime()
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return 0
    return Math.max(1, Math.round((end - start) / 60000))
  })()

  const continueHistory = () => {
    if (!historySession || historySession.status !== 'active') return
    setSession(historySession)
    setTurns(historyTurns)
    setGoal(historySession.goal || 'free')
    setTranscript('')
    setRecorderState('idle')
    setHistoryOpen(false)
    setError('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

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
        {realtimeError ? <Alert type="error" showIcon message={realtimeError} closable onClose={() => setRealtimeError('')} /> : null}

        <Card title="实时对话" data-testid="voice-realtime-card">
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            <Text type="secondary">持续采集麦克风，自动判断说完后回答；可以随时说话打断回答。</Text>
            <Space wrap>
              {realtimeState === 'idle' || realtimeState === 'stopped' ? (
                <Button type="primary" icon={<AudioOutlined />} onClick={() => void startRealtime()} data-testid="voice-realtime-start">开始实时对话</Button>
              ) : null}
              {realtimeState !== 'idle' && realtimeState !== 'stopped' ? (
                <>
                  <Button icon={realtimeState === 'paused' ? <PlayCircleOutlined /> : <PauseOutlined />} onClick={pauseRealtime} data-testid="voice-realtime-pause">{realtimeState === 'paused' ? '继续' : '暂停'}</Button>
                  <Button danger icon={<StopOutlined />} onClick={stopRealtime} data-testid="voice-realtime-stop">结束</Button>
                </>
              ) : null}
              <Tag icon={realtimeState === 'connecting' || realtimeState === 'thinking' ? <LoadingOutlined /> : undefined} color={realtimeState === 'listening' ? 'green' : realtimeState === 'speaking' ? 'blue' : realtimeState === 'thinking' ? 'gold' : 'default'}>
                {realtimeStatus || (realtimeState === 'connecting' ? '连接中' : realtimeState === 'listening' ? '正在听' : realtimeState === 'thinking' ? '思考中' : realtimeState === 'speaking' ? '回答中' : realtimeState === 'paused' ? '已暂停' : '未开始')}
              </Tag>
            </Space>
            {realtimeTranscript ? <Paragraph style={{ margin: 0 }}><Text type="secondary">你：</Text>{realtimeTranscript}</Paragraph> : null}
            {realtimeAnswer ? <Paragraph style={{ margin: 0, whiteSpace: 'pre-wrap' }}><Text type="secondary">回答：</Text>{realtimeAnswer}</Paragraph> : null}
            {realtimeCitations.length ? <Space wrap>{realtimeCitations.map((citation, index) => <Button key={citation.citation_id || index} size="small" onClick={() => setSource({ citation, index })}>[C{index + 1}] {citation.doc_name || '来源'}</Button>)}</Space> : null}
          </Space>
        </Card>

        {!session || session.status !== 'active' ? (
          <Card title="选择练习目标" data-testid="voice-start-card">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Select value={goal} onChange={setGoal} options={GOALS.map(item => ({ value: item.value, label: `${item.label} · ${item.description}` }))} style={{ maxWidth: 420, width: '100%' }} aria-label="练习目标" />
              <Button type="primary" icon={<AudioOutlined />} onClick={() => void begin()} data-testid="voice-start">开始练习</Button>
            </Space>
          </Card>
        ) : null}

        {session?.status === 'active' ? (
          <Card title={currentGoal?.label || '对话进行中'} data-testid="voice-session">
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
          <Card title="连续对话记录" data-testid="voice-turns">
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
              renderItem={item => <List.Item
                style={{ cursor: 'pointer' }}
                onClick={() => void loadHistory(item)}
                actions={[<Button
                  key="open"
                  type="text"
                  icon={<RightOutlined />}
                  aria-label="打开练习复盘"
                  data-testid="voice-history-open"
                  onClick={event => { event.stopPropagation(); void loadHistory(item) }}
                />]}
              >
                <List.Item.Meta
                  title={<Space size={8} wrap>
                    <Text strong>{item.first_transcript || GOALS.find(goalItem => goalItem.value === item.goal)?.label || '语音练习'}</Text>
                    <Tag color={item.status === 'active' ? 'processing' : item.status === 'completed' ? 'success' : 'default'}>
                      {item.status === 'active' ? '进行中' : item.status === 'completed' ? '已完成' : '已中止'}
                    </Tag>
                  </Space>}
                  description={`${GOALS.find(goalItem => goalItem.value === item.goal)?.label || '自由问答'} · ${item.turn_count || 0} 轮 · ${item.updated_at ? new Date(item.updated_at).toLocaleString() : ''}`}
                />
              </List.Item>}
            />
          )}
        </Card>
      </Space>
      <Drawer
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        title="练习复盘"
        placement={screens.md ? 'right' : 'bottom'}
        width={screens.md ? 720 : undefined}
        height={screens.md ? undefined : '92vh'}
        destroyOnClose
        data-testid="voice-review-drawer"
        extra={historySession?.status === 'active' ? <Button type="primary" onClick={continueHistory}>继续本次对话</Button> : null}
      >
        {historyLoading ? <div style={{ display: 'grid', placeItems: 'center', minHeight: 240 }}><Spin tip="正在加载完整记录" /></div> : historySession ? (
          <Space direction="vertical" size={24} style={{ width: '100%' }}>
            <div>
              <Space size={8} wrap>
                <Title level={4} style={{ margin: 0 }}>{GOALS.find(item => item.value === historySession.goal)?.label || '语音练习'}</Title>
                <Tag color={historySession.status === 'active' ? 'processing' : historySession.status === 'completed' ? 'success' : 'default'}>
                  {historySession.status === 'active' ? '进行中' : historySession.status === 'completed' ? '已完成' : '已中止'}
                </Tag>
              </Space>
              <Text type="secondary">
                {historySession.created_at ? new Date(historySession.created_at).toLocaleString() : ''}
                {historySession.ended_at ? ` 至 ${new Date(historySession.ended_at).toLocaleString()}` : ''}
              </Text>
            </div>

            <Row gutter={[12, 16]}>
              <Col xs={12} sm={6}><Statistic title="对话轮次" value={historyTurns.length} suffix="轮" /></Col>
              <Col xs={12} sm={6}><Statistic title="练习时长" value={historyDurationMinutes} suffix="分钟" /></Col>
              <Col xs={12} sm={6}><Statistic title="有来源回答" value={historyCitedTurns} suffix={`/ ${historyTurns.length}`} /></Col>
              <Col xs={12} sm={6}><Statistic title="使用个人记忆" value={historyMemoryTurns} suffix="轮" /></Col>
            </Row>

            <section>
              <Title level={5}>本次复盘</Title>
              <Space direction="vertical" size={6}>
                <Text>{historySession.summary || `本次共完成 ${historyCompletedTurns} 轮对话。`}</Text>
                <Text type={historyFailedTurns ? 'warning' : 'secondary'}>
                  {historyFailedTurns ? `有 ${historyFailedTurns} 轮未完成，可以回看当时的问题后重新提问。` : '所有轮次均已完成。'}
                </Text>
                <Text type={historyCitedTurns ? 'secondary' : 'warning'}>
                  {historyCitedTurns ? `${historyCitedTurns} 轮回答可以核对资料来源。` : '本次回答没有资料引用，复盘时建议重新核对知识库。'}
                </Text>
              </Space>
            </section>

            <section>
              <Title level={5}>完整对话</Title>
              {historyTurns.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="这次练习还没有形成对话记录" /> : (
                <List
                  dataSource={historyTurns}
                  split
                  renderItem={(turn, index) => (
                    <List.Item style={{ display: 'block', paddingInline: 0 }}>
                      <Space direction="vertical" size={10} style={{ width: '100%' }}>
                        <Space wrap>
                          <Text strong>第 {index + 1} 轮</Text>
                          <Text type="secondary">{turn.created_at ? new Date(turn.created_at).toLocaleTimeString() : ''}</Text>
                          {turn.status !== 'completed' || !turn.answer ? <Tag color="error">未完成</Tag> : null}
                        </Space>
                        <div style={{ borderLeft: '3px solid #1677ff', paddingLeft: 12 }}>
                          <Text type="secondary">你</Text>
                          <Paragraph style={{ margin: '4px 0 0', whiteSpace: 'pre-wrap' }}>{turn.transcript}</Paragraph>
                        </div>
                        <div style={{ borderLeft: '3px solid #52c41a', paddingLeft: 12 }}>
                          <Text type="secondary">回答</Text>
                          {turn.answer ? <Paragraph style={{ margin: '4px 0 0', whiteSpace: 'pre-wrap' }}>{turn.answer}</Paragraph> : <Text type="danger">本轮没有形成回答</Text>}
                        </div>
                        <Space wrap>
                          <Button size="small" icon={<CustomerServiceOutlined />} loading={playing === turn.id} onClick={() => void playAnswer(turn)} disabled={!turn.answer}>播放回答</Button>
                          {(turn.citations || []).map((citation, citeIndex) => (
                            <Button key={citation.citation_id || citeIndex} size="small" onClick={() => setSource({ citation, index: citeIndex })}>[C{citeIndex + 1}] {citation.doc_name || '核对来源'}</Button>
                          ))}
                          {(turn.memory_refs || []).map((memory, memoryIndex) => <Tag key={memory.memory_id || memoryIndex} color="purple">[M{memoryIndex + 1}] {memory.content || '个人记忆'}</Tag>)}
                        </Space>
                      </Space>
                    </List.Item>
                  )}
                />
              )}
            </section>
          </Space>
        ) : null}
      </Drawer>
      <SourceReader open={Boolean(source)} citation={source?.citation || null} index={source?.index || 0} kbId={kbId} onClose={() => setSource(null)} />
    </div>
  )
}
