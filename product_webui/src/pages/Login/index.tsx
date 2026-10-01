import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Card, Form, Input, Typography, message } from 'antd'
import { authApi, setTokens } from '../../api'
import { PRODUCT_NAME } from '../../constants'

const { Title, Text } = Typography

export default function LoginPage() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [submitting, setSubmitting] = useState(false)

  const submit = async (values: { email: string; password: string }) => {
    setSubmitting(true)
    try {
      const result = mode === 'register'
        ? await authApi.register(values.email, values.password)
        : await authApi.login(values.email, values.password)
      setTokens(result.access_token)
      message.success(mode === 'register' ? '注册成功' : '已登录')
      navigate('/', { replace: true })
    } catch (error) {
      message.error(typeof error === 'string' ? error : '登录失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f5f6fa' }}>
      <Card style={{ width: 400 }} data-testid="login-card">
        <Title level={3} style={{ marginTop: 0 }}>{PRODUCT_NAME}</Title>
        <Text type="secondary">{mode === 'login' ? '使用邮箱登录你的知识库' : '注册后只看得到自己的知识库'}</Text>
        <Form layout="vertical" style={{ marginTop: 24 }} onFinish={submit}>
          <Form.Item name="email" label="邮箱" rules={[{ required: true, type: 'email', message: '请输入邮箱' }]}>
            <Input autoComplete="username" data-testid="email" />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true, min: 8, pattern: /[A-Za-z]/, message: '密码至少 8 位，且包含字母' }]}>
            <Input.Password autoComplete={mode === 'login' ? 'current-password' : 'new-password'} data-testid="password" />
          </Form.Item>
          <Button type="primary" htmlType="submit" block loading={submitting} data-testid="auth-submit">
            {mode === 'login' ? '登录' : '注册'}
          </Button>
        </Form>
        <Button type="link" style={{ paddingLeft: 0, marginTop: 8 }} data-testid="auth-switch" onClick={() => setMode(current => current === 'login' ? 'register' : 'login')}>
          {mode === 'login' ? '没有账号，去注册' : '已有账号，去登录'}
        </Button>
      </Card>
    </div>
  )
}
