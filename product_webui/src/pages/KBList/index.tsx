import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Layout, Typography, Button, Card, Row, Col, Modal, Form, Input,
  Empty, Spin, Space, Tag, Dropdown, message,
} from 'antd'
import {
  PlusOutlined, DatabaseOutlined, MoreOutlined,
  FileTextOutlined, DeleteOutlined, EditOutlined,
} from '@ant-design/icons'
import { authApi, kbApi } from '../../api'
import AccountDrawer from '../../components/AccountDrawer'
import DeepCheckView from '../../components/DeepCheckView'
import { PRODUCT_NAME, PRODUCT_SUBTITLE, PRODUCT_TAG } from '../../constants'
import dayjs from 'dayjs'

const { Header, Content } = Layout
const { Title, Text } = Typography

export default function KBListPage() {
  const navigate = useNavigate()
  const [kbs, setKbs]       = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [editing, setEditing] = useState<any>(null)
  const [form] = Form.useForm()
  const [editForm] = Form.useForm()
  const [account, setAccount] = useState('')
  const [authOn, setAuthOn] = useState(false)
  const [purgeJob, setPurgeJob] = useState<any>(null)
  const [accountOpen, setAccountOpen] = useState(false)
  const [purgeCheck, setPurgeCheck] = useState<any>(null)

  const load = () => {
    setLoading(true)
    kbApi.list().then((r: any) => setKbs(r.items ?? [])).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    authApi.status().then((status: any) => {
      setAuthOn(Boolean(status?.enabled))
      if (status?.enabled) {
        authApi.me().then((user: any) => setAccount(user?.email || '')).catch(() => setAccount(''))
      }
    }).catch(() => setAuthOn(false))
  }, [])

  useEffect(() => {
    const finished = ['succeeded', 'failed', 'timeout', 'cancelled', 'consistency_failed']
    if (!purgeJob?.job_id || finished.includes(purgeJob.status)) return
    const timer = window.setInterval(() => {
      kbApi.purgeJob(purgeJob.job_id).then(setPurgeJob).catch(() => undefined)
    }, 2000)
    return () => window.clearInterval(timer)
  }, [purgeJob?.job_id, purgeJob?.status])

  const handleCreate = async () => {
    const vals = await form.validateFields()
    await kbApi.create(vals)
    message.success('知识库创建成功')
    setCreating(false)
    form.resetFields()
    load()
  }

  const openEdit = (kb: any, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditing(kb)
    editForm.setFieldsValue({ name: kb.name, description: kb.description || '' })
  }

  const handleEdit = async () => {
    const vals = await editForm.validateFields()
    await kbApi.update(editing.id, vals)
    message.success('知识库已更新')
    setEditing(null)
    editForm.resetFields()
    load()
  }

  const handleDelete = async (kb: any, e: React.MouseEvent) => {
    e.stopPropagation()
    Modal.confirm({
      title: `确认删除「${kb.name}」？`,
      content: '知识库会先从列表移出，并在后台清理文档、向量、图谱和源文件。失败后可以重试。',
      okType: 'danger',
      okText: '确认删除',
      onOk: async () => {
        const result = await kbApi.delete(kb.id)
        message.success('已移入回收站，正在清理')
        if (result?.job_id) setPurgeJob(result)
        load()
      },
    })
  }

  return (
    <Layout style={{ minHeight: '100vh', background: '#f5f6fa' }}>
      <Header
        style={{
          background: '#fff',
          borderBottom: '1px solid #f0f0f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 16px',
        }}
      >
        <Space>
          <DatabaseOutlined style={{ fontSize: 20, color: '#7c5cfc' }} />
          <Title level={4} style={{ margin: 0 }}>
            {PRODUCT_SUBTITLE ? `${PRODUCT_NAME} · ${PRODUCT_SUBTITLE}` : PRODUCT_NAME}
          </Title>
        </Space>
        <Space>
          {authOn && account ? <Text type="secondary" style={{ maxWidth: 140 }} ellipsis>{account}</Text> : null}
          {authOn ? (
            <Button data-testid="account-open" aria-label="打开账户" onClick={() => setAccountOpen(true)}>账户</Button>
          ) : null}
          {PRODUCT_TAG ? <Tag>{PRODUCT_TAG}</Tag> : null}
        </Space>
      </Header>

      <Content style={{ padding: '32px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <div>
            <Title level={4} style={{ margin: 0 }} data-testid="kb-home">知识库</Title>
            <Text type="secondary">按主题整理个人资料，并在资料之间查找联系</Text>
          </div>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreating(true)}
          >
            新建知识库
          </Button>
        </div>

        {purgeJob ? (
          <Card style={{ marginBottom: 16 }}>
            <Space>
              <Text>删除进度 {purgeJob.progress ?? 0}%</Text>
              <Tag>
                {purgeJob.status === 'succeeded'
                  ? '已完成'
                  : purgeJob.status === 'failed' || purgeJob.status === 'timeout'
                    ? '失败'
                    : purgeJob.status === 'consistency_failed'
                      ? '校验未通过'
                      : purgeJob.status === 'cancelled'
                        ? '已取消'
                        : '进行中'}
              </Tag>
              {purgeJob.stage || purgeJob.step ? <Text type="secondary">{purgeJob.stage || purgeJob.step}</Text> : null}
              {purgeJob.error ? <Text type="danger" style={{ overflowWrap: 'anywhere' }}>{purgeJob.error}</Text> : null}
              {purgeJob.status === 'succeeded' ? (
                <Button size="small" onClick={async () => setPurgeCheck(await kbApi.deepCheck(purgeJob.job_id))}>深度检查</Button>
              ) : null}
              {purgeJob.status === 'failed' || purgeJob.status === 'timeout' || purgeJob.status === 'consistency_failed' ? (
                <Button
                  size="small"
                  onClick={async () => setPurgeJob(await kbApi.retryPurge(purgeJob.job_id))}
                >
                  重试
                </Button>
              ) : null}
            </Space>
          </Card>
        ) : null}
        {purgeCheck ? <Card style={{ marginBottom: 16 }}><DeepCheckView report={purgeCheck} /></Card> : null}

        {loading ? (
          <div style={{ textAlign: 'center', padding: 80 }}><Spin size="large" /></div>
        ) : kbs.length === 0 ? (
          <Empty description="暂无知识库，点击右上角新建" style={{ marginTop: 80 }} />
        ) : (
          <Row gutter={[16, 16]}>
            {kbs.map((kb: any) => (
              <Col key={kb.id} xs={24} sm={12} md={8} lg={6}>
                <Card
                  hoverable
                  onClick={() => navigate(`/kb/${kb.id}/documents`)}
                  style={{ borderRadius: 8, cursor: 'pointer' }}
                  extra={
                    <Dropdown
                      menu={{
                        items: [
                          {
                            key: 'edit',
                            icon: <EditOutlined />,
                            label: '编辑名称',
                            onClick: ({ domEvent }) => openEdit(kb, domEvent as any),
                          },
                          {
                            key: 'delete',
                            icon: <DeleteOutlined />,
                            label: '删除知识库',
                            danger: true,
                            onClick: ({ domEvent }) => handleDelete(kb, domEvent as any),
                          },
                        ],
                      }}
                      trigger={['click']}
                    >
                      <MoreOutlined onClick={(e) => e.stopPropagation()} />
                    </Dropdown>
                  }
                >
                  <Space direction="vertical" style={{ width: '100%' }} size={8}>
                    <Space>
                      <div
                        style={{
                          width: 36, height: 36, borderRadius: 8,
                          background: '#f0ebff',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}
                      >
                        <FileTextOutlined style={{ color: '#7c5cfc', fontSize: 18 }} />
                      </div>
                      <Text strong style={{ fontSize: 15 }}>{kb.name}</Text>
                    </Space>
                    <Text type="secondary" style={{ fontSize: 12 }} ellipsis>
                      {kb.description || '暂无描述'}
                    </Text>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      创建于 {dayjs(kb.created_at).format('YYYY-MM-DD')}
                    </Text>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        )}
      </Content>

      <Modal
        title="新建知识库"
        open={creating}
        onOk={handleCreate}
        onCancel={() => { setCreating(false); form.resetFields() }}
        okText="创建"
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item name="name" label="知识库名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="例：项目资料库" />
          </Form.Item>
          <Form.Item name="description" label="描述（可选）">
            <Input.TextArea rows={3} placeholder="描述该库覆盖的资料范围" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="编辑知识库"
        open={!!editing}
        onOk={handleEdit}
        onCancel={() => { setEditing(null); editForm.resetFields() }}
        okText="保存"
      >
        <Form form={editForm} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item name="name" label="知识库名称" rules={[{ required: true, message: '请输入名称' }]}>
            <Input placeholder="知识库名称" />
          </Form.Item>
          <Form.Item name="description" label="描述（可选）">
            <Input.TextArea rows={3} placeholder="描述该库覆盖的资料范围" />
          </Form.Item>
        </Form>
      </Modal>

      <AccountDrawer open={accountOpen} onClose={() => setAccountOpen(false)} />
    </Layout>
  )
}
