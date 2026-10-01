import { useEffect, useState } from 'react'
import { Drawer, Spin, Typography } from 'antd'
import { docApi } from '../api'
import type { Citation, DocumentSource } from '../types'

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

export function SourceDrawer({
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
  const [settled, setSettled] = useState(false)

  useEffect(() => {
    if (!remote || !citation) {
      setSource(null)
      setMissing(false)
      setSettled(false)
      return
    }
    let alive = true
    setSettled(false)
    setMissing(false)
    setSource(null)
    docApi.source(citation.document_id || '', {
      kb_id: kbId || '',
      version_id: citation.version_id,
      chunk_id: citation.chunk_id,
      unit_id: citation.unit_id,
    }).then(value => {
      if (!alive) return
      setSource(value)
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

  const place = citePlace(source?.unit) || citePlace(citation?.location) || citePlace(citation)
  const excerpt = source?.matched_chunk?.excerpt || citation?.excerpt || ''
  const parent = source?.parent_context || citation?.parent_content || ''
  const showParent = Boolean(parent && parent !== excerpt)
  const name = source?.doc_name || citation?.doc_name || '未命名资料'
  const kind = kindLabel(source?.source_kind)
  const precise = Boolean(place)

  return (
    <Drawer
      title={citation ? `来源 ${citation.citation_id || `[${index}]`}` : '来源'}
      open={open}
      onClose={onClose}
      width={Math.min(440, window.innerWidth - 16)}
      data-testid="source-drawer"
    >
      {remote && !settled ? <div data-testid="source-loading"><Spin /></div> : null}
      {missing ? <div data-testid="source-missing">来源不可用。资料可能已删除或已更新。</div> : null}
      {!missing && (!remote || settled) && citation ? (
        <div style={{ overflowWrap: 'anywhere' }}>
          <Text strong>{name}</Text>
          {kind ? <Paragraph type="secondary">{kind}</Paragraph> : null}
          {precise ? <Paragraph data-testid="source-place">{place}</Paragraph> : <div data-testid="source-unlocated">暂无精确位置</div>}
          <Text type="secondary">命中内容</Text>
          <Paragraph>{excerpt || source?.source_text || '这条引用没有摘录'}</Paragraph>
          {showParent ? (
            <>
              <Text type="secondary">补充上下文</Text>
              <Paragraph>{parent}</Paragraph>
            </>
          ) : null}
          {source?.preview?.available && source.preview.url ? null : null}
        </div>
      ) : null}
    </Drawer>
  )
}
