import { useEffect, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import { Typography, Button, Space, List, Switch, Input, Modal, Tag, Empty, message, Popconfirm } from 'antd'
import { memoryApi } from '../../api'
import type { MemoryCandidate, MemoryItem } from '../../types'

const { Title, Paragraph, Text } = Typography
const { TextArea } = Input

export default function MemoryPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [candidates, setCandidates] = useState<MemoryCandidate[]>([])
  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [editing, setEditing] = useState<MemoryItem | null>(null)
  const [draft, setDraft] = useState('')

  const load = () => {
    memoryApi.listCandidates(kbId).then(page => setCandidates(page.items || []))
    memoryApi.list().then(page => setMemories(page.items || []))
  }

  useEffect(() => { load() }, [kbId])

  return (
    <div style={{ padding: 16, maxWidth: 860 }} data-testid="memory-page">
      <Title level={4}>记忆</Title>
      <Paragraph type="secondary">
        模型只能提出记忆候选。确认后才会用于回答，并标记为 [M1]。文档证据仍用 [C1]。删除知识库不会删除已确认的个人记忆。
      </Paragraph>
      <Title level={5}>待确认候选</Title>
      {candidates.length === 0 ? <Empty description="没有待确认的记忆" /> : (
        <List
          dataSource={candidates}
          renderItem={item => (
            <List.Item
              data-testid="memory-candidate"
              actions={[
                <Button key="ok" type="link" data-testid="memory-accept" onClick={() => memoryApi.accept(item.id, kbId).then(() => { message.success('已保存为记忆'); load() })}>确认</Button>,
                <Button key="no" type="link" data-testid="memory-reject" onClick={() => memoryApi.reject(item.id, kbId).then(() => { message.success('已忽略'); load() })}>忽略</Button>,
              ]}
            >
              <List.Item.Meta title={item.content} description={<Tag>{item.category || 'other'}</Tag>} />
            </List.Item>
          )}
        />
      )}
      <Title level={5} style={{ marginTop: 24 }}>已确认记忆</Title>
      {memories.length === 0 ? <Empty description="还没有个人记忆" /> : (
        <List
          dataSource={memories}
          renderItem={item => (
            <List.Item
              data-testid="memory-item"
              actions={[
                <Switch
                  key="on"
                  data-testid="memory-enabled"
                  checked={item.enabled !== false}
                  onChange={(enabled) => memoryApi.patch(item.id, { enabled }).then(() => load())}
                />,
                <Button key="edit" type="link" data-testid="memory-edit" onClick={() => { setEditing(item); setDraft(item.content) }}>编辑</Button>,
                <Popconfirm key="del" title="删除这条记忆？" onConfirm={() => memoryApi.remove(item.id).then(() => load())}>
                  <Button type="link" danger data-testid="memory-delete">删除</Button>
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={item.content}
                description={<Space><Tag>{item.category || 'other'}</Tag><Text type="secondary">{item.enabled === false ? '已关闭，不参与回答' : '已启用'}</Text></Space>}
              />
            </List.Item>
          )}
        />
      )}
      <Modal
        title="编辑记忆"
        open={Boolean(editing)}
        onCancel={() => setEditing(null)}
        onOk={() => {
          if (!editing) return
          memoryApi.patch(editing.id, { content: draft }).then(() => { setEditing(null); load() })
        }}
      >
        <TextArea rows={4} value={draft} onChange={event => setDraft(event.target.value)} data-testid="memory-edit-input" />
      </Modal>
    </div>
  )
}
