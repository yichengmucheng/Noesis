import { useEffect, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import { Typography, Table, Button, Tabs, Tag, Space, Empty, Modal, Input, message } from 'antd'
import { CheckOutlined, CloseOutlined } from '@ant-design/icons'
import { auditApi, kbApi } from '../../api'
import dayjs from 'dayjs'

const { Title } = Typography

const statusColor: Record<string, string> = { 待审核: 'gold', 已通过: 'green', 已拒绝: 'red' }

export default function AuditPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [tab, setTab] = useState('document')
  const [tables, setTables] = useState<Record<string, any[]>>({ document: [], qa: [], chunk: [] })
  const [counts, setCounts] = useState({ document: 0, qa: 0, chunk: 0 })
  const [loading, setLoading] = useState(false)
  const [rejecting, setRejecting] = useState<any>(null)
  const [comment, setComment] = useState('')
  const [needComment, setNeedComment] = useState(true)

  const load = (itemType = tab) => {
    setLoading(true)
    auditApi.list(kbId, { item_type: itemType })
      .then((r: any) => {
        const rows = r.items ?? []
        setTables(prev => ({ ...prev, [itemType]: rows }))
        setCounts(prev => ({ ...prev, [itemType]: r.total ?? rows.length }))
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    kbApi.getSettings(kbId).then((s: any) => setNeedComment(s.audit_reject_comment !== false))
    ;(['document', 'qa', 'chunk'] as const).forEach(type => {
      auditApi.list(kbId, { item_type: type }).then((r: any) => {
        const rows = r.items ?? []
        setCounts(prev => ({ ...prev, [type]: r.total ?? rows.length }))
        setTables(prev => ({ ...prev, [type]: rows }))
      })
    })
  }, [kbId])

  const approve = async (row: any) => {
    await auditApi.review(row.id, kbId, 'approved')
    message.success('已通过')
    load()
  }

  const reject = async () => {
    if (needComment && !comment.trim()) {
      message.error('拒绝时需要填写原因')
      return
    }
    await auditApi.review(rejecting.id, kbId, 'rejected', comment.trim())
    message.success('已拒绝')
    setRejecting(null)
    setComment('')
    load()
  }

  const columns = [
    { title: '名称', dataIndex: 'name', ellipsis: true },
    { title: '操作类型', dataIndex: 'action_type', width: 100 },
    { title: '内容', dataIndex: 'chunk_info', ellipsis: true },
    { title: '更新时间', dataIndex: 'updated_at', width: 160, render: (t: string) => t ? dayjs(t).format('YYYY-MM-DD HH:mm') : '-' },
    { title: '状态', dataIndex: 'status', width: 90, render: (s: string) => <Tag color={statusColor[s] || 'default'}>{s || '待审核'}</Tag> },
    {
      title: '操作',
      width: 160,
      render: (_: any, r: any) => (
        <Space>
          <Button size="small" type="primary" icon={<CheckOutlined />} onClick={() => approve(r)}>通过</Button>
          <Button size="small" danger icon={<CloseOutlined />} onClick={() => { setRejecting(r); setComment(r.comment || '') }}>拒绝</Button>
        </Space>
      ),
    },
  ]

  const tableFor = (itemType: string) => (
    <Table rowKey="id" columns={columns} dataSource={tables[itemType] || []} loading={loading && tab === itemType}
      locale={{ emptyText: <Empty description="暂无待审核内容" /> }} />
  )

  return (
    <div style={{ padding: 24 }}>
      <Title level={4} style={{ marginBottom: 16 }}>审核中心</Title>
      <Tabs
        activeKey={tab}
        onChange={(key) => { setTab(key); load(key) }}
        items={[
          { key: 'document', label: `文件审核 (${counts.document})`, children: tableFor('document') },
          { key: 'qa', label: `问答审核 (${counts.qa})`, children: tableFor('qa') },
          { key: 'chunk', label: `文本块审核 (${counts.chunk})`, children: tableFor('chunk') },
        ]}
      />
      <Modal
        title="拒绝原因"
        open={!!rejecting}
        okText="确认拒绝"
        okButtonProps={{ danger: true }}
        onCancel={() => setRejecting(null)}
        onOk={reject}
      >
        <Input.TextArea rows={4} value={comment} onChange={e => setComment(e.target.value)} placeholder="说明拒绝原因" />
      </Modal>
    </div>
  )
}
