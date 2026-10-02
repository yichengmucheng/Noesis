import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { Drawer, Spin, Typography } from 'antd'
import { docApi } from '../api'
import type { Citation, DocumentSource, SourceContent } from '../types'

const { Text, Paragraph } = Typography

export function citePlace(citation: Citation | DocumentSource['unit'] | null | undefined): string {
  if (!citation) return ''
  const parts: string[] = []
  const page = 'page_number' in citation ? citation.page_number : undefined
  const legacyPage = citation && 'page_num' in citation ? (citation as Citation).page_num : undefined
  const pageNumber = page ?? legacyPage
  if (pageNumber) parts.push(`第${pageNumber}页`)
  if (citation.slide_number) parts.push(`第${citation.slide_number}张幻灯片`)
  const section = Array.isArray(citation.section_path) ? citation.section_path.filter(Boolean).join(' / ') : ''
  if (section) parts.push(section)
  if (citation.sheet_name) parts.push(String(citation.sheet_name))
  if (citation.cell_range) parts.push(String(citation.cell_range))
  const start = 'line_start' in citation ? citation.line_start : undefined
  const end = 'line_end' in citation ? citation.line_end : undefined
  if (start) parts.push(end && end !== start ? `第${start}-${end}行` : `第${start}行`)
  return parts.join(' · ')
}

function kindLabel(kind?: string): string {
  const names: Record<string, string> = {
    pdf: 'PDF', docx: 'Word', pptx: 'PPT', xlsx: 'Excel', markdown: 'Markdown', text: '文本',
  }
  return names[kind || ''] || kind || ''
}

function bboxStyle(bbox?: number[] | null): CSSProperties | null {
  if (!bbox || bbox.length < 4) return null
  const [x0, y0, x1, y1] = bbox
  const pageW = 612
  const pageH = 792
  const left = Math.max(0, Math.min(x0, x1))
  const right = Math.min(pageW, Math.max(x0, x1))
  const bottom = Math.max(0, Math.min(y0, y1))
  const top = Math.min(pageH, Math.max(y0, y1))
  if (right <= left || top <= bottom) return null
  return {
    position: 'absolute',
    left: `${(left / pageW) * 100}%`,
    top: `${((pageH - top) / pageH) * 100}%`,
    width: `${((right - left) / pageW) * 100}%`,
    height: `${((top - bottom) / pageH) * 100}%`,
    background: 'rgba(250, 173, 20, 0.45)',
    border: '2px solid #d48806',
    pointerEvents: 'none',
  }
}

export function SourceReader({
  open,
  citation,
  index,
  kbId,
  onClose,
}: {
  open: boolean
  citation: Citation | null
  index: number
  kbId?: string
  onClose: () => void
}) {
  const remote = Boolean(open && citation?.document_id && kbId)
  const [missing, setMissing] = useState(false)
  const [source, setSource] = useState<DocumentSource | null>(null)
  const [reading, setReading] = useState<SourceContent | null>(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [settled, setSettled] = useState(false)
  const narrow = typeof window !== 'undefined' && window.innerWidth <= 480

  useEffect(() => {
    if (!remote || !citation) {
      setSource(null)
      setReading(null)
      setMissing(false)
      setSettled(false)
      setPreviewUrl(current => {
        if (current) URL.revokeObjectURL(current)
        return ''
      })
      return
    }
    let alive = true
    setSettled(false)
    setMissing(false)
    setSource(null)
    setReading(null)
    const params = {
      kb_id: kbId || '',
      version_id: citation.version_id,
      chunk_id: citation.chunk_id,
      unit_id: citation.unit_id,
    }
    docApi.source(citation.document_id || '', params).then(value => {
      if (!alive) return
      setSource(value)
      return docApi.content(citation.document_id || '', params).then(body => {
        if (alive) setReading(body)
      }).catch(() => undefined)
    }).catch(() => {
      if (!alive) return
      setMissing(true)
    }).finally(() => {
      if (!alive) return
      setSettled(true)
    })
    return () => {
      alive = false
    }
  }, [remote, citation, kbId])

  useEffect(() => {
    if (!source?.preview?.available || !citation?.document_id || !kbId) return
    let alive = true
    let current = ''
    docApi.preview(citation.document_id, { kb_id: kbId, version_id: citation.version_id }).then(blob => {
      if (!alive) return
      current = URL.createObjectURL(blob)
      setPreviewUrl(current)
    }).catch(() => undefined)
    return () => {
      alive = false
      if (current) URL.revokeObjectURL(current)
    }
  }, [source, citation, kbId])

  const place = citePlace(source?.unit) || citePlace(citation?.location) || citePlace(citation)
  const excerpt = source?.matched_chunk?.excerpt || citation?.excerpt || ''
  const parent = source?.parent_context || citation?.parent_content || ''
  const showParent = Boolean(parent && parent !== excerpt)
  const name = source?.doc_name || reading?.doc_name || citation?.doc_name || '未命名资料'
  const kind = source?.source_kind || reading?.source_kind || ''
  const precise = Boolean(place)
  const page = source?.unit?.page_number || citation?.page_number || citation?.location?.page_number
  const bbox = source?.unit?.bbox || citation?.bbox || citation?.location?.bbox
  const highlight = bboxStyle(bbox)
  const lineStart = source?.unit?.line_start || citation?.line_start || citation?.location?.line_start || 0
  const lineEnd = source?.unit?.line_end || citation?.line_end || citation?.location?.line_end || lineStart
  const text = reading?.text || ''
  const lines = useMemo(() => (text ? text.split(/\r?\n/) : []), [text])
  const structured = kind === 'docx' || kind === 'pptx' || kind === 'xlsx' || reading?.layout === 'structured'
  const frameUrl = previewUrl && page ? `${previewUrl}#page=${page}` : previewUrl

  return (
    <Drawer
      title={citation ? `来源 ${citation.citation_id || `[${index}]`}` : '来源'}
      open={open}
      onClose={onClose}
      width={narrow ? '100%' : Math.min(760, window.innerWidth - 24)}
      height={narrow ? '100%' : undefined}
      placement={narrow ? 'bottom' : 'right'}
      data-testid="source-drawer"
      styles={{ body: { padding: 16 } }}
    >
      <div data-testid="source-reader" style={{ minHeight: narrow ? '70vh' : undefined }}>
        {remote && !settled ? <div data-testid="source-loading"><Spin /></div> : null}
        {missing ? <div data-testid="source-missing">来源不可用。资料可能已删除或已更新。</div> : null}
        {!missing && (!remote || settled) && citation ? (
          <div style={{ overflowWrap: 'anywhere' }}>
            <Text strong>{name}</Text>
            {kindLabel(kind) ? <Paragraph type="secondary">{kindLabel(kind)}</Paragraph> : null}
            {precise ? <Paragraph data-testid="source-place">{place}</Paragraph> : (
              <div data-testid="source-unlocated">已找到来源，但暂无精确位置</div>
            )}
            {kind === 'pdf' && frameUrl ? (
              <div data-testid="pdf-page" style={{ position: 'relative', height: 480, marginBottom: 12 }}>
                <iframe title={name} src={frameUrl} style={{ width: '100%', height: '100%', border: 0 }} />
                {highlight ? <div data-testid="pdf-highlight" style={highlight} /> : null}
              </div>
            ) : null}
            {kind === 'pdf' && !frameUrl && page ? <Paragraph data-testid="pdf-page">第{page}页</Paragraph> : null}
            {kind === 'pdf' && highlight && !frameUrl ? <div data-testid="pdf-highlight" style={{ ...highlight, position: 'relative', height: 48 }} /> : null}
            {lines.length ? (
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                {lines.map((line, offset) => {
                  const number = offset + 1
                  const marked = Boolean(lineStart) && number >= Number(lineStart) && number <= Number(lineEnd || lineStart)
                  return (
                    <div key={number} data-testid={marked ? 'text-highlight' : undefined} style={marked ? { background: '#fff1b8' } : undefined}>
                      {line || ' '}
                    </div>
                  )
                })}
              </pre>
            ) : null}
            {structured ? (
              <div data-testid="structured-reading">
                <Paragraph type="secondary">{reading?.notice || '这是结构化阅读视图，不是原始 Office 排版。'}</Paragraph>
                {(reading?.units || []).slice(0, 40).map((unit, offset) => (
                  <Paragraph key={String(unit.unit_id || offset)}>{citePlace(unit as DocumentSource['unit'])} {String(unit.content || '')}</Paragraph>
                ))}
              </div>
            ) : null}
            <Text type="secondary">命中内容</Text>
            <Paragraph>{excerpt || source?.source_text || '这条引用没有摘录'}</Paragraph>
            {showParent ? (
              <>
                <Text type="secondary">补充上下文</Text>
                <Paragraph>{parent}</Paragraph>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </Drawer>
  )
}

export const SourceDrawer = SourceReader
