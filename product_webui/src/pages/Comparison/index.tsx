import { useEffect, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Typography, Table, Button, Tabs, Tag, Space, Empty, Modal, Form, Select, Input, Drawer, List, Alert,
} from 'antd'
import { PlusOutlined } from '@ant-design/icons'
import { comparisonApi, docApi } from '../../api'
import dayjs from 'dayjs'

const { Title, Text, Paragraph } = Typography

const severityColor: Record<string, string> = { 高: 'red', 中: 'orange', 低: 'blue' }

function DiffList({ items }: { items: any[] }) {
  if (!items?.length) return <Empty description="没有发现差异" />
  return (
    <List
      dataSource={items}
      renderItem={(item: any) => (
        <List.Item>
          <div style={{ width: '100%' }}>
            <Space style={{ marginBottom: 6 }}>
              <Tag>{item.type}</Tag>
              {item.severity ? <Tag color={severityColor[item.severity] || 'default'}>{item.severity}</Tag> : null}
              {item.index ? <Text type="secondary">第 {item.index} 块</Text> : null}
            </Space>
            {item.question ? <Paragraph strong style={{ marginBottom: 6 }}>{item.question}</Paragraph> : null}
            {item.old ? <Paragraph style={{ marginBottom: 4, background: '#fff1f0', padding: 8 }}>{item.old}</Paragraph> : null}
            {item.new ? <Paragraph style={{ marginBottom: 0, background: '#f6ffed', padding: 8 }}>{item.new}</Paragraph> : null}
          </div>
        </List.Item>
      )}
    />
  )
}

export default function ComparisonPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [tasks, setTasks] = useState<any[]>([])
  const [updates, setUpdates] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [docs, setDocs] = useState<any[]>([])
  const [creating, setCreating] = useState(false)
  const [viewing, setViewing] = useState<any>(null)
  const [form] = Form.useForm()

  const load = () => {
    setLoading(true)
    comparisonApi.list(kbId).then((r: any) => setTasks(r.items ?? [])).finally(() => setLoading(false))
    comparisonApi.updates(kbId).then((r: any) => setUpdates(r.items ?? [])).catch(() => setUpdates([]))
  }

  useEffect(() => {
    load()
    docApi.list({ kb_id: kbId, page: 1, page_size: 100 }).then((r: any) => setDocs(r.items ?? []))
  }, [kbId])

  const qaRows = tasks.flatMap(task => (task.qa_diffs || []).map((diff: any, index: number) => ({
    ...diff, key: `${task.task_id}-qa-${index}`, task: `${task.doc_name_old || task.doc_id_old} → ${task.doc_name_new || task.doc_id_new}`,
  })))
  const chunkRows = tasks.flatMap(task => (task.chunk_diffs || []).map((diff: any, index: number) => ({
    ...diff, key: `${task.task_id}-chunk-${index}`, task: `${task.doc_name_old || task.doc_id_old} → ${task.doc_name_new || task.doc_id_new}`,
  })))

  const columns = [
    { title: '新文件', dataIndex: 'doc_name_new', ellipsis: true, render: (v: string, r: any) => v || r.doc_id_new },
    { title: '旧文件', dataIndex: 'doc_name_old', ellipsis: true, render: (v: string, r: any) => v || r.doc_id_old },
    { title: '比对原因', dataIndex: 'reason' },
    {
      title: '差异',
      render: (_: any, r: any) => (
        <Text type="secondary">
          文件 {r.summary?.file_changes ?? 0} · 文本块 {r.summary?.chunk_changes ?? 0} · 问答 {r.summary?.qa_changes ?? 0}
        </Text>
      ),
    },
    { title: '任务状态', dataIndex: 'status', render: (s: string) => <Tag color={s === 'done' ? 'green' : 'default'}>{s === 'done' ? '已完成' : s}</Tag> },
    { title: '时间', dataIndex: 'created_at', render: (t: string) => t ? dayjs(t).format('MM-DD HH:mm') : '-' },
    {
      title: '操作',
      render: (_: any, r: any) => (
        <Button type="link" size="small" onClick={async () => {
          const detail = await comparisonApi.get(r.task_id)
          setViewing(detail)
        }}>查看比对</Button>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16, alignItems: 'flex-end' }}>
        <div>
          <Title level={4} style={{ margin: 0 }}>知识比对</Title>
          <Text type="secondary">对比两份资料，核对说法差异</Text>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreating(true)}>新建比对任务</Button>
      </div>
      <Tabs
        items={[
          { key: 'file', label: `文件比对 (${tasks.length})`, children: (
            <Table rowKey="task_id" columns={columns} dataSource={tasks} loading={loading}
              locale={{ emptyText: <Empty description="暂无比对任务" /> }} />
          )},
          { key: 'qa', label: `问答比对 (${qaRows.length})`, children: (
            <Table rowKey="key" dataSource={qaRows} locale={{ emptyText: <Empty description="暂无问答差异" /> }}
              columns={[
                { title: '任务', dataIndex: 'task', ellipsis: true },
                { title: '类型', dataIndex: 'type', width: 120 },
                { title: '问题', dataIndex: 'question', ellipsis: true },
                { title: '旧', dataIndex: 'old', ellipsis: true },
                { title: '新', dataIndex: 'new', ellipsis: true },
              ]} />
          )},
          { key: 'chunk', label: `文本块比对 (${chunkRows.length})`, children: (
            <Table rowKey="key" dataSource={chunkRows} locale={{ emptyText: <Empty description="暂无文本块差异" /> }}
              columns={[
                { title: '任务', dataIndex: 'task', ellipsis: true },
                { title: '类型', dataIndex: 'type', width: 80 },
                { title: '序号', dataIndex: 'index', width: 70 },
                { title: '旧文本', dataIndex: 'old', ellipsis: true },
                { title: '新文本', dataIndex: 'new', ellipsis: true },
              ]} />
          )},
          { key: 'updates', label: `知识更新 (${updates.length})`, children: (
            <Table
              rowKey="change_set_id"
              dataSource={updates}
              locale={{ emptyText: <Empty description="暂无自动知识更新记录" /> }}
              columns={[
                { title: '文档', dataIndex: 'document_id', ellipsis: true },
                { title: '变化', render: (_: any, r: any) => `${r.summary?.changed ?? 0} 个文本块 · ${r.summary?.fact_changes ?? 0} 条事实` },
                { title: '状态', dataIndex: 'status', render: (s: string) => <Tag color={s === 'published' ? 'green' : 'blue'}>{s === 'published' ? '已发布' : '待处理'}</Tag> },
                { title: '操作', render: (_: any, r: any) => <Button type="link" onClick={async () => setViewing(await comparisonApi.update(kbId, r.change_set_id))}>查看更新</Button> },
              ]}
            />
          )},
        ]}
      />

      <Modal
        title="新建比对任务"
        open={creating}
        okText="开始比对"
        onCancel={() => { setCreating(false); form.resetFields() }}
        onOk={async () => {
          const vals = await form.validateFields()
          await comparisonApi.create({ kb_id: kbId, ...vals })
          setCreating(false)
          form.resetFields()
          load()
        }}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="doc_id_old" label="旧文档" rules={[{ required: true, message: '请选择旧文档' }]}>
            <Select options={docs.map(item => ({ value: item.id, label: item.name }))} />
          </Form.Item>
          <Form.Item name="doc_id_new" label="新文档" rules={[{ required: true, message: '请选择新文档' }]}>
            <Select options={docs.map(item => ({ value: item.id, label: item.name }))} />
          </Form.Item>
          <Form.Item name="reason" label="比对原因">
            <Input placeholder="例如：版本修订" />
          </Form.Item>
        </Form>
      </Modal>

      <Drawer title="比对结果" open={!!viewing} width={720} onClose={() => setViewing(null)}>
        {viewing && (
          <>
            <Paragraph>
              {viewing.doc_name_old} → {viewing.doc_name_new}
              {viewing.reason ? ` · ${viewing.reason}` : ''}
            </Paragraph>
            {viewing.impact ? (
              <Alert
                type="info"
                showIcon
                style={{ marginBottom: 12 }}
                message="更新影响范围"
                description={`变化文本块 ${viewing.impact.changed_chunks ?? 0} · 新增 ${viewing.impact.added_chunks ?? 0} · 修改 ${viewing.impact.modified_chunks ?? 0} · 删除 ${viewing.impact.deleted_chunks ?? 0} · 事实 ${viewing.impact.fact_changes ?? 0} · 问答 ${viewing.impact.qa_changes ?? 0}`}
              />
            ) : null}
            <Tabs items={[
              { key: 'file', label: `文件差异 (${viewing.file_diffs?.length || 0})`, children: <DiffList items={viewing.file_diffs || []} /> },
              { key: 'chunk', label: `文本块 (${viewing.chunk_diffs?.length || 0})`, children: <DiffList items={viewing.chunk_diffs || []} /> },
              { key: 'qa', label: `问答 (${viewing.qa_diffs?.length || 0})`, children: <DiffList items={viewing.qa_diffs || []} /> },
              { key: 'facts', label: `事实 (${viewing.fact_diffs?.length || 0})`, children: (
                <List
                  dataSource={viewing.fact_diffs || []}
                  locale={{ emptyText: '没有实体或关系变化' }}
                  renderItem={(item: any) => <List.Item><Space><Tag color={item.status === 'active' ? 'green' : 'orange'}>{item.type}</Tag><Text>{item.subject} {item.predicate} {item.object}</Text></Space></List.Item>}
                />
              )},
            ]} />
            {viewing.change_set_id && viewing.status !== 'published' ? (
              <Button type="primary" onClick={async () => {
                const next = await comparisonApi.publish(kbId, viewing.change_set_id)
                setViewing(next)
                load()
              }}>发布知识更新</Button>
            ) : null}
          </>
        )}
      </Drawer>
    </div>
  )
}
