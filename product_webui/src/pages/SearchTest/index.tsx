import { useState, useEffect, useRef } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Typography, Input, Button, Select, Slider, Switch, Space,
  Card, Tag, Empty, Spin, Divider, Row, Col, message, Alert, Collapse,
} from 'antd'
import { SearchOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { chatApi, kbApi } from '../../api'
import { RETRIEVAL_MODE_OPTIONS } from '../../constants'
import type { SearchResponse, SearchResult } from '../../types'

const { Title, Text, Paragraph } = Typography

function hitPlace(chunk: SearchResult): string {
  const parts: string[] = []
  const page = chunk.page_number ?? chunk.page_num
  if (page) parts.push(`第${page}页`)
  if (chunk.slide_number) parts.push(`第${chunk.slide_number}张幻灯片`)
  const section = Array.isArray(chunk.section_path) ? chunk.section_path.filter(Boolean).join(' / ') : ''
  if (section) parts.push(section)
  if (chunk.sheet_name) parts.push(String(chunk.sheet_name))
  if (chunk.cell_range) parts.push(String(chunk.cell_range))
  return parts.join(' · ')
}

const MODE_OPTIONS = RETRIEVAL_MODE_OPTIONS.map(({ value, label, desc }) => ({ value, label, desc }))
const MODE_DESC = Object.fromEntries(RETRIEVAL_MODE_OPTIONS.map(item => [item.value, item.desc]))

const SOURCE_COLOR: Record<string, string> = {
  vector: 'blue', keyword: 'orange', hybrid: 'purple', mix: 'magenta', graph: 'green', tree: 'cyan',
}

export default function SearchTestPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [query, setQuery]   = useState('')
  const [mode, setMode]     = useState('mix')
  const [topK, setTopK]     = useState(10)
  const [ratio, setRatio]   = useState(0.5)
  const [rerank, setRerank] = useState(false)
  const [elapsed, setElapsed] = useState<number | null>(null)
  const [threshold, setThreshold] = useState(0.2)
  const [graphEnabled, setGraphEnabled] = useState(true)
  const [results, setResults] = useState<SearchResult[]>([])
  const [meta, setMeta] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  const requestSeq = useRef(0)

  useEffect(() => {
    kbApi.getSettings(kbId).then((s: any) => {
      if (s.retrieval_mode) setMode(s.retrieval_mode)
      if (s.top_k) setTopK(Math.min(30, Math.max(1, Number(s.top_k))))
      if (s.similarity_ratio != null) setRatio(Number(s.similarity_ratio))
      setRerank(!!s.rerank_enabled)
      if (s.similarity_threshold != null) setThreshold(Number(s.similarity_threshold))
      setGraphEnabled(s.graph_enabled !== false)
    }).catch(() => {})
  }, [kbId])

  const run = async () => {
    if (!query.trim()) return
    const seq = ++requestSeq.current
    setLoading(true)
    setResults([])
    setMeta(null)
    setElapsed(null)
    const started = performance.now()
    setHistory(h => [query, ...h.slice(0, 9)])
    try {
      const r = await chatApi.searchTest({
        query, kb_id: kbId, retrieval_mode: mode,
        top_k: topK, similarity_ratio: ratio, enable_rerank: rerank,
        similarity_threshold: threshold,
      })
      if (seq !== requestSeq.current) return
      const chunks = Array.isArray(r?.chunks) ? r.chunks : []
      setMeta(r)
      setResults(chunks)
      setElapsed(Math.round(performance.now() - started))
    } catch (e: any) {
      if (seq !== requestSeq.current) return
      setResults([])
      setMeta(null)
      message.error(typeof e === 'string' ? e : '检索失败，请确认文档已入库完成')
    } finally {
      if (seq === requestSeq.current) setLoading(false)
    }
  }

  return (
    <div style={{ padding: 16, minWidth: 0 }}>
      <Title level={4} style={{ marginBottom: 16 }}>检索测试</Title>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card title="输入检索词" style={{ borderRadius: 8 }}>
            <Input.TextArea
              rows={6}
              placeholder="例如：这份资料的核心结论是什么"
              value={query}
              onChange={e => setQuery(e.target.value)}
              showCount maxLength={500}
              onPressEnter={e => { if (!e.shiftKey) { e.preventDefault(); run() } }}
            />
            <Space style={{ marginTop: 12, width: '100%', justifyContent: 'flex-end' }}>
              <Button
                data-testid="search-run"
                type="primary"
                icon={<ThunderboltOutlined />}
                onClick={run}
                loading={loading}
              >
                测试
              </Button>
            </Space>

            <Divider orientation="left" plain style={{ fontSize: 12 }}>参数配置</Divider>
            <Space direction="vertical" style={{ width: '100%' }} size={12}>
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>检索模式</Text>
                <Select
                  value={mode} onChange={setMode}
                  options={MODE_OPTIONS} style={{ width: '100%', marginTop: 4 }}
                />
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 6 }}>
                  {MODE_DESC[mode]}
                </Text>
                {!graphEnabled && (mode === 'mix' || mode === 'graph') ? (
                  <Text type="warning" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                    知识库已关闭图谱多跳，本次按向量检索执行
                  </Text>
                ) : null}
              </div>
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>返回数量 Top-{topK}</Text>
                <Slider min={1} max={30} value={topK} onChange={setTopK} />
              </div>
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  向量:关键词 = {Math.round(ratio * 100)}%:{Math.round((1 - ratio) * 100)}%
                  {mode !== 'hybrid' ? '（仅混合检索使用）' : ''}
                </Text>
                <Slider min={0} max={1} step={0.1} value={ratio} onChange={setRatio} disabled={mode !== 'hybrid'} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <Text type="secondary" style={{ fontSize: 12 }}>重排（交叉编码器）</Text>
                  <div style={{ fontSize: 12, color: '#8c8c8c' }}>用原始问题对文档名、标题路径和小块正文做交叉编码重排</div>
                </div>
                <Switch checked={rerank} onChange={setRerank} size="small" />
              </div>
            </Space>

            {history.length > 0 && (
              <>
                <Divider orientation="left" plain style={{ fontSize: 12 }}>测试历史</Divider>
                <Space wrap>
                  {history.map((q, i) => (
                    <Tag
                      key={i}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setQuery(q)}
                    >{q}</Tag>
                  ))}
                </Space>
              </>
            )}
          </Card>
        </Col>

        {/* 右：结果区 */}
        <Col xs={24} lg={14}>
          <Card
            title={
              results.length > 0
                ? <Space><SearchOutlined />测试结果 <Tag>{results.length} 条</Tag></Space>
                : '测试结果'
            }
            style={{ borderRadius: 8, minHeight: 400 }}
          >
            {meta?.status === 'index_rebuild_required' ? (
              <div data-testid="index-mismatch">
                <Alert type="error" showIcon message="需要重新构建索引" description={meta.empty_reason || meta.rebuild_reason || '当前模型与索引不一致，已停止使用旧索引。'} />
              </div>
            ) : null}
            {meta?.query_plan ? (
              <div data-testid="search-plan" style={{ marginBottom: 12, fontSize: 12, color: '#595959', overflowWrap: 'anywhere' }}>
                路由 {meta.query_plan.route} · {meta.query_plan.reason}
                {elapsed != null ? ` · ${elapsed} ms` : ''}
              </div>
            ) : null}
            {meta && meta.status !== 'index_rebuild_required' ? (
              <Collapse
                style={{ marginBottom: 12 }}
                items={[{
                  key: 'diag',
                  label: '检索诊断',
                  children: (
                    <div data-testid="search-diagnostics" style={{ fontSize: 12, overflowWrap: 'anywhere' }}>
                      <div>独立问题 {meta.query_plan?.standalone_query || '—'}</div>
                      <div>问法变体 {(meta.query_plan?.variants || []).join('；') || '—'}</div>
                      <div>HyDE {meta.query_plan?.hyde_text ? '已运行，只用于召回' : '未运行'}</div>
                      <div>子问题 {(meta.query_plan?.subquestions || []).join('；') || '—'}</div>
                      <div>
                        候选 稠密 {meta.retrieval?.dense_top ?? '—'} / 关键词 {meta.retrieval?.bm25_top ?? '—'} / 图谱 {meta.retrieval?.graph_top ?? '—'} / RRF {meta.retrieval?.rrf ?? '—'}
                      </div>
                      <div>重排池 {meta.retrieval?.rerank_pool ?? meta.rerank?.pool_size ?? '—'} · 最终小块 {meta.retrieval?.children ?? results.length} · 父块 {meta.retrieval?.parents ?? '—'}</div>
                      <div>重排模型 {meta.rerank?.model || '未配置'}{meta.rerank?.applied ? ' · 已执行' : ''}</div>
                      {meta.timings ? <div>耗时 {Object.entries(meta.timings).map(([key, value]) => `${key} ${value}ms`).join(' · ')}</div> : null}
                      {results.map((chunk, i) => (
                        <div key={chunk.chunk_id || i} style={{ marginTop: 8 }}>
                          #{i + 1}
                          {chunk.channel ? ` · 渠道 ${chunk.channel}` : ''}
                          {chunk.channel_rank != null ? ` · 渠道名次 ${chunk.channel_rank}` : ''}
                          {chunk.path_id ? ` · 路径 ${chunk.path_id}` : ''}
                          {chunk.path ? ` · ${chunk.path}` : ''}
                          {chunk.parent_id ? ` · 父块 ${chunk.parent_id}` : ''}
                          {chunk.admission_reason || chunk.admission_score != null ? ` · admission ${chunk.admission_reason || chunk.admission_score}` : ''}
                        </div>
                      ))}
                    </div>
                  ),
                }]}
              />
            ) : null}
            {meta?.rerank?.warning ? (
              <Alert type="warning" showIcon style={{ marginBottom: 12 }} message={meta.rerank.warning} />
            ) : null}
            {loading ? (
              <div data-testid="search-loading" style={{ textAlign: 'center', padding: 60 }}><Spin size="large" /></div>
            ) : meta?.status === 'index_rebuild_required' ? null : results.length === 0 ? (
              <div data-testid="search-empty" data-empty-reason={meta?.empty_reason || ''}>
                <Empty description={meta ? (meta.empty_reason || '没有达到相关度要求的内容') : '测试结果将在这里展示'} style={{ marginTop: 40 }} />
              </div>
            ) : (
              <Space direction="vertical" style={{ width: '100%' }} size={12}>
                {results.map((chunk, i) => {
                  const source = chunk.source || mode
                  const score = chunk.score ?? 0
                  return (
                  <Card
                    key={chunk.chunk_id ?? i}
                    size="small"
                    style={{ borderRadius: 8, border: '1px solid #f0f0f0' }}
                    title={
                      <Space>
                        <Text type="secondary" style={{ fontSize: 12 }}>#{i + 1}</Text>
                        <Tag color={SOURCE_COLOR[source] ?? 'default'} style={{ fontSize: 11 }}>
                          {source}
                        </Tag>
                        <Text type="secondary" style={{ fontSize: 12 }}>{chunk.doc_name}{hitPlace(chunk) ? ` · ${hitPlace(chunk)}` : ''}</Text>
                        {chunk.channel ? <Tag style={{ fontSize: 11 }}>{chunk.channel}</Tag> : null}
                        {chunk.hop != null ? <Text type="secondary" style={{ fontSize: 11 }}>{chunk.hop} 跳</Text> : null}
                        {chunk.reranked ? <Tag style={{ fontSize: 11 }}>重排</Tag> : null}
                      </Space>
                    }
                    extra={
                      <Tag color={score > 0.8 ? 'green' : score > 0.5 ? 'orange' : 'red'}>
                        {score.toFixed(4)}
                      </Tag>
                    }
                  >
                    <Paragraph
                      ellipsis={{ rows: 4, expandable: true }}
                      style={{ fontSize: 13, margin: 0 }}
                    >
                      {chunk.excerpt}
                    </Paragraph>
                  </Card>
                  )
                })}
              </Space>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}
