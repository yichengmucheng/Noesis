import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Drawer, Form, Input, List, Space, Tag, Typography, message } from 'antd'
import dayjs from 'dayjs'
import { authApi, clearTokens } from '../api'
import type { DeviceSession } from '../types'

const { Text, Paragraph } = Typography

function when(value?: string) {
  if (!value) return '—'
  const parsed = dayjs(value)
  return parsed.isValid() ? parsed.format('YYYY-MM-DD HH:mm') : value
}

export default function AccountDrawer({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [sessions, setSessions] = useState<DeviceSession[]>([])
  const [loading, setLoading] = useState(false)
  const [passwordOpen, setPasswordOpen] = useState(false)
  const [form] = Form.useForm()

  const load = () => {
    setLoading(true)
    Promise.all([
      authApi.me().catch(() => null),
      authApi.sessions().catch(() => ({ items: [] as DeviceSession[] })),
    ]).then(([user, page]) => {
      setEmail(user?.email || '')
      setSessions(page.items || [])
    }).finally(() => setLoading(false))
  }

  useEffect(() => {
    if (open) load()
  }, [open])

  const leave = async (all: boolean) => {
    try {
      if (all) await authApi.logoutAll()
      else await authApi.logout()
    } catch {
      /* 会话已失效时直接离开 */
    }
    clearTokens()
    onClose()
    navigate('/login')
  }

  const revoke = async (session: DeviceSession) => {
    await authApi.revokeSession(session.session_id)
    if (session.is_current) {
      clearTokens()
      onClose()
      navigate('/login')
      return
    }
    message.success('已撤销该设备')
    load()
  }

  return (
    <Drawer
      title="账户"
      open={open}
      onClose={onClose}
      width={Math.min(420, window.innerWidth - 24)}
      destroyOnClose
    >
      <Space direction="vertical" style={{ width: '100%' }} size={16}>
        <div>
          <Text type="secondary">当前邮箱</Text>
          <Paragraph style={{ marginBottom: 0, overflowWrap: 'anywhere' }} data-testid="account-email">
            {email || '—'}
          </Paragraph>
        </div>
        <Button data-testid="change-password" onClick={() => setPasswordOpen(true)}>修改密码</Button>
        {passwordOpen ? (
          <Form
            form={form}
            layout="vertical"
            onFinish={async (values) => {
              await authApi.changePassword(values.old_password, values.new_password)
              message.success('密码已更新，请重新登录')
              clearTokens()
              navigate('/login')
            }}
          >
            <Form.Item name="old_password" label="当前密码" rules={[{ required: true }]}>
              <Input.Password />
            </Form.Item>
            <Form.Item
              name="new_password"
              label="新密码"
              rules={[{ required: true, min: 8, pattern: /[A-Za-z]/, message: '至少 8 位且包含字母' }]}
            >
              <Input.Password />
            </Form.Item>
            <Button type="primary" htmlType="submit">保存</Button>
          </Form>
        ) : null}
        <div data-testid="session-list">
          <Text strong>当前设备</Text>
          <List
            loading={loading}
            dataSource={sessions}
            locale={{ emptyText: '没有可管理的设备会话' }}
            renderItem={(session) => (
              <List.Item
                actions={[
                  <Button
                    key="revoke"
                    size="small"
                    danger
                    data-testid="session-revoke"
                    data-current={session.is_current ? '1' : '0'}
                    aria-label={session.is_current ? '撤销当前设备' : '撤销此设备'}
                    onClick={() => revoke(session)}
                  >
                    撤销
                  </Button>,
                ]}
              >
                <List.Item.Meta
                  title={
                    <Space wrap>
                      <span style={{ overflowWrap: 'anywhere' }}>{session.device_label || '未知设备'}</span>
                      {session.is_current ? <Tag>当前设备</Tag> : null}
                    </Space>
                  }
                  description={
                    <span style={{ overflowWrap: 'anywhere' }}>
                      创建 {when(session.created_at)} · 最后使用 {when(session.last_used_at)}
                    </span>
                  }
                />
              </List.Item>
            )}
          />
        </div>
        <Space wrap>
          <Button onClick={() => leave(true)}>全部退出</Button>
          <Button onClick={() => leave(false)}>退出</Button>
        </Space>
      </Space>
    </Drawer>
  )
}
