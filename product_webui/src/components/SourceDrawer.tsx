import { Drawer, Typography } from 'antd'
import type { Citation, SourceReaderProps } from '../types'

const { Text, Paragraph } = Typography

export function citePlace(citation: Citation): string {
  const parts: string[] = []
  const page = citation.page_number ?? citation.page_num
  if (page) parts.push(`第${page}页`)
  if (citation.slide_number) parts.push(`第${citation.slide_number}张幻灯片`)
  const section = Array.isArray(citation.section_path) ? citation.section_path.filter(Boolean).join(' / ') : ''
  if (section) parts.push(section)
  if (citation.sheet_name) parts.push(String(citation.sheet_name))
  if (citation.cell_range) parts.push(String(citation.cell_range))
  return parts.join(' · ')
}

export function SourceDrawer({
  open,
  citation,
  index,
  onClose,
}: {
  open: boolean
  citation: Citation | null
  index: number
  onClose: () => void
}) {
  const place = citation ? citePlace(citation) : ''
  const parent = citation?.parent_content || ''
  const excerpt = citation?.excerpt || ''
  const showParent = Boolean(parent && parent !== excerpt)
  return (
    <Drawer
      title={citation ? `来源 [${index}]` : '来源'}
      open={open}
      onClose={onClose}
      width={Math.min(440, window.innerWidth - 16)}
      data-testid="source-drawer"
    >
      {citation ? (
        <div style={{ overflowWrap: 'anywhere' }}>
          <Text strong>{citation.doc_name || '未命名资料'}</Text>
          {place ? <Paragraph type="secondary">{place}</Paragraph> : null}
          <Text type="secondary">命中内容</Text>
          <Paragraph>{excerpt || '这条引用没有摘录'}</Paragraph>
          {showParent ? (
            <>
              <Text type="secondary">补充上下文</Text>
              <Paragraph>{parent}</Paragraph>
            </>
          ) : null}
        </div>
      ) : null}
    </Drawer>
  )
}

export type { SourceReaderProps }
