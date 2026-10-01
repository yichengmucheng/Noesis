import { Typography } from 'antd'
import { STORE_LABEL } from '../constants'
import type { DeepCheckReport } from '../types'

const { Text, Paragraph } = Typography

export default function DeepCheckView({ report }: { report: DeepCheckReport | null }) {
  if (!report) return null
  const stores = report.stores || {}
  const leftover = Object.entries(stores).filter(([, value]) => (value?.count || 0) > 0)
  return (
    <div data-testid="deep-check" style={{ overflowWrap: 'anywhere' }}>
      <Paragraph style={{ marginBottom: 8 }}>
        {report.passed ? '删除后的资料、索引和关联都已清理干净。' : '删除后仍有内容没有清理干净。'}
      </Paragraph>
      {leftover.length === 0 ? (
        <Text type="secondary">没有发现残留。</Text>
      ) : (
        leftover.map(([name, value]) => (
          <Paragraph key={name} style={{ marginBottom: 4 }}>
            {STORE_LABEL[name] || '其他记录'}仍有 {value.count} 处未清理
          </Paragraph>
        ))
      )}
    </div>
  )
}
