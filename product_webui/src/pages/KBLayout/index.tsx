import { useEffect, useState } from 'react'
import { Outlet, useNavigate, useParams, useLocation } from 'react-router-dom'
import { Layout, Menu, Button, Typography, Spin, Grid, Drawer, Tooltip } from 'antd'
import {
  FileTextOutlined, MessageOutlined, SearchOutlined, ApartmentOutlined,
  SettingOutlined, DiffOutlined, AuditOutlined, MenuFoldOutlined, MenuUnfoldOutlined,
  UserOutlined, ToolOutlined,
} from '@ant-design/icons'
import { kbApi } from '../../api'
import AccountDrawer from '../../components/AccountDrawer'

const { Sider, Content } = Layout
const { Text } = Typography

const PRIMARY = [
  { key: 'documents', icon: <FileTextOutlined />, label: '资料' },
  { key: 'chat', icon: <MessageOutlined />, label: '问答' },
]

const ADVANCED = [
  { key: 'search', icon: <SearchOutlined />, label: '检索测试' },
  { key: 'graph', icon: <ApartmentOutlined />, label: '知识图谱' },
  { key: 'comparison', icon: <DiffOutlined />, label: '知识比对' },
  { key: 'audit', icon: <AuditOutlined />, label: '审核中心' },
  { key: 'settings', icon: <SettingOutlined />, label: '高级设置' },
]

export default function KBLayout() {
  const { kbId } = useParams<{ kbId: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const screens = Grid.useBreakpoint()
  const isMobile = !screens.md
  const [kbName, setKbName] = useState('')
  const [loading, setLoading] = useState(true)
  const [collapsed, setCollapsed] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [accountOpen, setAccountOpen] = useState(false)

  const section = location.pathname.split('/').pop() || 'documents'

  useEffect(() => {
    if (!kbId) return
    setLoading(true)
    kbApi.get(kbId)
      .then((kb) => setKbName(kb?.name || kbId))
      .catch(() => navigate('/'))
      .finally(() => setLoading(false))
  }, [kbId])

  useEffect(() => {
    if (screens.lg) setCollapsed(false)
    else if (screens.md) setCollapsed(true)
  }, [screens.lg, screens.md])

  const go = (key: string) => {
    navigate(`/kb/${kbId}/${key}`)
    setAdvancedOpen(false)
  }

  if (loading) {
    return (
      <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Spin size="large" />
      </div>
    )
  }

  const advancedMenu = (
    <Menu
      mode="inline"
      selectedKeys={[section]}
      style={{ border: 'none' }}
      items={ADVANCED}
      onClick={({ key }) => go(key)}
    />
  )

  return (
    <Layout style={{ minHeight: '100vh', background: '#f5f6fa' }}>
      {!isMobile ? (
        <Sider
          width={200}
          collapsedWidth={64}
          collapsed={collapsed}
          theme="light"
          style={{ borderRight: '1px solid #f0f0f0', background: '#fff' }}
        >
          <div style={{ padding: collapsed ? 8 : 16, borderBottom: '1px solid #f0f0f0' }}>
            <Text strong ellipsis style={{ display: 'block', overflowWrap: 'anywhere' }}>
              {collapsed ? '库' : kbName}
            </Text>
          </div>
          <Menu
            mode="inline"
            selectedKeys={PRIMARY.some(item => item.key === section) ? [section] : []}
            style={{ border: 'none' }}
            items={PRIMARY.map(item => ({
              ...item,
              label: <span data-testid={`nav-${item.key}`}>{item.label}</span>,
            }))}
            onClick={({ key }) => go(key)}
          />
          <Menu
            mode="inline"
            selectedKeys={ADVANCED.some(item => item.key === section) ? [section] : []}
            style={{ border: 'none' }}
            items={[{
              key: 'advanced',
              icon: <ToolOutlined />,
              label: <span data-testid="nav-advanced">高级工具</span>,
              children: ADVANCED.map(item => ({
                ...item,
                label: <span data-testid={`nav-${item.key}`}>{item.label}</span>,
              })),
            }]}
            onClick={({ key }) => { if (key !== 'advanced') go(key) }}
          />
          <div style={{ padding: 8 }}>
            <Tooltip title={collapsed ? '展开导航' : '收起导航'}>
              <Button
                type="text"
                aria-label={collapsed ? '展开导航' : '收起导航'}
                data-testid="nav-toggle"
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setCollapsed(value => !value)}
              />
            </Tooltip>
          </div>
        </Sider>
      ) : null}

      <Layout style={{ minWidth: 0 }}>
        <div style={{
          height: 48,
          background: '#fff',
          borderBottom: '1px solid #f0f0f0',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px',
          gap: 8,
        }}>
          <Text strong ellipsis style={{ minWidth: 0 }}>{kbName}</Text>
          <Tooltip title="账户">
            <Button
              type="text"
              aria-label="打开账户"
              data-testid="account-open"
              icon={<UserOutlined />}
              onClick={() => setAccountOpen(true)}
            />
          </Tooltip>
        </div>
        <Content style={{ overflow: 'auto', minWidth: 0, paddingBottom: isMobile ? 64 : 0 }}>
          <Outlet context={{ kbId, kbName }} />
        </Content>
      </Layout>

      {isMobile ? (
        <div style={{
          position: 'fixed',
          left: 0,
          right: 0,
          bottom: 0,
          height: 56,
          background: '#fff',
          borderTop: '1px solid #f0f0f0',
          display: 'flex',
          zIndex: 20,
        }}>
          {PRIMARY.map(item => (
            <Button
              key={item.key}
              type="text"
              data-testid={`nav-${item.key}`}
              aria-label={item.label}
              icon={item.icon}
              onClick={() => go(item.key)}
              style={{ flex: 1, height: 56, color: section === item.key ? '#1a1a2e' : '#8c8c8c' }}
            >
              {item.label}
            </Button>
          ))}
          <Button
            type="text"
            data-testid="nav-advanced"
            aria-label="高级工具"
            icon={<ToolOutlined />}
            onClick={() => setAdvancedOpen(true)}
            style={{ flex: 1, height: 56 }}
          >
            更多
          </Button>
        </div>
      ) : null}

      <Drawer
        title="高级工具"
        placement="bottom"
        height={360}
        open={isMobile && advancedOpen}
        onClose={() => setAdvancedOpen(false)}
      >
        {advancedMenu}
      </Drawer>
      <AccountDrawer open={accountOpen} onClose={() => setAccountOpen(false)} />
    </Layout>
  )
}
