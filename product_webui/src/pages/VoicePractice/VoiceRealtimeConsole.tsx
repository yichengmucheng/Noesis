import { Collapse, Progress, Space, Statistic, Table, Tag, Tooltip, Typography } from 'antd'
import { AudioOutlined, CustomerServiceOutlined, InfoCircleOutlined } from '@ant-design/icons'
import type { VoicePracticeTurn, VoiceTurnDiagnostics } from '../../types'
import './voice-realtime.css'

const { Paragraph, Text } = Typography

type RealtimeState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'paused' | 'stopped'

interface Props {
  state: RealtimeState
  status: string
  transcript: string
  answer: string
  micLevel: number
  outputLevel: number
  micSamples: number[]
  outputSamples: number[]
  diagnostics: VoiceTurnDiagnostics
  recentTurns: VoicePracticeTurn[]
}

const milliseconds = (value?: number) => Number.isFinite(value) ? `${Math.round(value as number)} ms` : '—'
const decibels = (value?: number) => Number.isFinite(value) ? `${(value as number).toFixed(1)} dB` : '—'

function Waveform({ samples, tone }: { samples: number[]; tone: 'input' | 'output' }) {
  const values = samples.length ? samples.slice(-36) : Array.from({ length: 36 }, () => 0.02)
  return (
    <div className={`voice-waveform voice-waveform-${tone}`} aria-hidden="true">
      {values.map((value, index) => <span key={index} style={{ height: `${Math.max(4, Math.min(100, value * 100))}%` }} />)}
    </div>
  )
}

function bottleneck(metrics: VoiceTurnDiagnostics) {
  const retrieval = metrics.retrieval_ms || 0
  const stages = [
    ['ASR 定稿', metrics.asr_finalize_ms || 0],
    ['资料检索', retrieval],
    ['LLM 生成', Math.max(0, (metrics.llm_ttft_ms || 0) - retrieval)],
    ['TTS 首包', metrics.tts_ttfb_ms || 0],
  ] as const
  const slowest = stages.reduce((current, item) => item[1] > current[1] ? item : current, stages[0])
  return slowest[1] > 0 ? `当前主要耗时：${slowest[0]} ${Math.round(slowest[1])} ms` : '完成一轮对话后显示瓶颈'
}

export default function VoiceRealtimeConsole({
  state, status, transcript, answer, micLevel, outputLevel, micSamples, outputSamples, diagnostics, recentTurns,
}: Props) {
  const revisions = diagnostics.asr_revisions || 0
  const interim = diagnostics.asr_interim_count || 0
  const stability = interim ? Math.max(0, Math.round((1 - revisions / interim) * 100)) : undefined
  const stageRows = [
    { key: 'vad', label: 'VAD 断句', value: diagnostics.vad_tail_ms },
    { key: 'asr', label: 'ASR 定稿', value: diagnostics.asr_finalize_ms },
    { key: 'retrieval', label: '资料检索', value: diagnostics.retrieval_ms },
    { key: 'llm', label: 'LLM 首响', value: diagnostics.llm_ttft_ms },
    { key: 'tts', label: 'TTS 首包', value: diagnostics.tts_ttfb_ms },
    { key: 'e2e', label: '端到端', value: diagnostics.e2e_ms },
  ]

  return (
    <div className="voice-console" data-testid="voice-realtime-console">
      <div className="voice-console-header">
        <div>
          <Text strong>实时通话</Text>
          <Text type="secondary" className="voice-console-subtitle">知识库检索、个人记忆与语音链路保持同一会话</Text>
        </div>
        <Tag color={state === 'listening' ? 'success' : state === 'speaking' ? 'blue' : state === 'thinking' ? 'gold' : 'default'}>
          {status || '未开始'}
        </Tag>
      </div>

      <div className="voice-latency-strip" data-testid="voice-latency-strip">
        <Statistic title="LLM 首响" value={milliseconds(diagnostics.llm_ttft_ms)} />
        <Statistic title="TTS 首包" value={milliseconds(diagnostics.tts_ttfb_ms)} />
        <Statistic title="端到端" value={milliseconds(diagnostics.e2e_ms)} />
        <div className="voice-bottleneck"><Text type="secondary">{bottleneck(diagnostics)}</Text></div>
      </div>

      <div className="voice-channel-grid">
        <section className={`voice-channel voice-channel-input ${state === 'listening' ? 'is-active' : ''}`}>
          <div className="voice-channel-heading">
            <Space><AudioOutlined /><Text strong>麦克风输入</Text></Space>
            <Text type="secondary">{decibels(diagnostics.input_peak_db)}</Text>
          </div>
          <div className="voice-meter-row">
            <Progress percent={Math.round(micLevel * 100)} showInfo={false} strokeColor="#16a085" trailColor="#e7eeec" />
            <Text type="secondary">{state === 'listening' ? '正在聆听' : state === 'thinking' ? '正在处理' : '等待输入'}</Text>
          </div>
          <Waveform samples={micSamples} tone="input" />
          <div className="voice-live-copy">
            <Text type="secondary">你</Text>
            <Paragraph>{transcript || '说话后，增量识别文字会出现在这里'}</Paragraph>
          </div>
        </section>

        <section className={`voice-channel voice-channel-output ${state === 'speaking' ? 'is-active' : ''}`}>
          <div className="voice-channel-heading">
            <Space><CustomerServiceOutlined /><Text strong>TTS 输出</Text></Space>
            <Text type="secondary">{state === 'speaking' ? '流式播放' : '等待回答'}</Text>
          </div>
          <div className="voice-meter-row">
            <Progress percent={Math.round(outputLevel * 100)} showInfo={false} strokeColor="#df6b83" trailColor="#f2e7ea" />
            <Text type="secondary">{state === 'speaking' ? '正在输出' : state === 'thinking' ? '准备语音' : '静音'}</Text>
          </div>
          <Waveform samples={outputSamples} tone="output" />
          <div className="voice-live-copy">
            <Text type="secondary">回答</Text>
            <Paragraph>{answer || '回答生成后会边合成边播放'}</Paragraph>
          </div>
        </section>
      </div>

      <Collapse
        ghost
        className="voice-diagnostics"
        items={[{
          key: 'diagnostics',
          label: <Space><span>通话诊断</span><Tag>{stageRows.filter(item => Number.isFinite(item.value)).length}/6 环节</Tag></Space>,
          children: (
            <Space direction="vertical" size={20} style={{ width: '100%' }}>
              <div className="voice-stage-grid">
                {stageRows.map(item => <div className="voice-stage" key={item.key}><Text type="secondary">{item.label}</Text><strong>{milliseconds(item.value)}</strong></div>)}
              </div>
              <div>
                <div className="voice-section-title">
                  <Text strong>ASR 质量与鲁棒性</Text>
                  <Tooltip title="CER/WER 必须用人工标注文本进行离线评测，在线会话只展示可观测的质量代理指标。"><InfoCircleOutlined /></Tooltip>
                </div>
                <div className="voice-quality-grid">
                  <Statistic title="首个增量" value={milliseconds(diagnostics.asr_first_partial_ms)} />
                  <Statistic title="转写稳定度" value={stability === undefined ? '—' : `${stability}%`} />
                  <Statistic title="增量修订" value={revisions} suffix={interim ? `/ ${interim}` : ''} />
                  <Statistic title="识别失败" value={diagnostics.asr_errors || 0} suffix="次" />
                  <Statistic title="连接恢复" value={diagnostics.reconnects || 0} suffix="次" />
                  <Statistic title="CER / WER" value="待标注评测" />
                </div>
              </div>
              {recentTurns.length ? (
                <div>
                  <div className="voice-section-title"><Text strong>最近轮次</Text></div>
                  <Table
                    size="small"
                    pagination={false}
                    scroll={{ x: 620 }}
                    rowKey="id"
                    dataSource={recentTurns.slice(-8).reverse()}
                    columns={[
                      { title: '问题', dataIndex: 'transcript', ellipsis: true, width: 220 },
                      { title: 'ASR', render: (_, row) => milliseconds(row.diagnostics?.asr_finalize_ms), width: 90 },
                      { title: '检索', render: (_, row) => milliseconds(row.diagnostics?.retrieval_ms), width: 90 },
                      { title: 'LLM', render: (_, row) => milliseconds(row.diagnostics?.llm_ttft_ms), width: 90 },
                      { title: 'TTS', render: (_, row) => milliseconds(row.diagnostics?.tts_ttfb_ms), width: 90 },
                      { title: '端到端', render: (_, row) => milliseconds(row.diagnostics?.e2e_ms), width: 100 },
                    ]}
                  />
                </div>
              ) : null}
            </Space>
          ),
        }]}
      />
    </div>
  )
}
