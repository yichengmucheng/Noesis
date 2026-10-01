import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { authApi, ensureSession, isLoggedIn } from './api'
import KBListPage from './pages/KBList'
import LoginPage from './pages/Login'
import KBLayout from './pages/KBLayout'
import DocumentsPage from './pages/Documents'
import SearchTestPage from './pages/SearchTest'
import KBSettingsPage from './pages/KBSettings'
import ComparisonPage from './pages/Comparison'
import AuditPage from './pages/Audit'
import GraphPage from './pages/Graph'
import ChatPage from './pages/Chat'

const basename = import.meta.env.BASE_URL.replace(/\/$/, '')

function AuthGate({ children }: { children: React.ReactNode }) {
  const [phase, setPhase] = useState<'boot' | 'open'>('boot')
  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      await ensureSession()
      if (cancelled) return
      let enabled = false
      try {
        const status = await authApi.status()
        enabled = Boolean(status?.enabled)
      } catch {
        enabled = false
      }
      if (cancelled) return
      const onLogin = location.pathname.startsWith('/login')
      if (enabled && !isLoggedIn() && !onLogin) {
        navigate('/login', { replace: true })
        return
      }
      if (enabled && isLoggedIn() && onLogin) {
        navigate('/', { replace: true })
        return
      }
      setPhase('open')
    })()
    return () => { cancelled = true }
  }, [location.pathname, navigate])

  if (phase !== 'open') return null
  return <>{children}</>
}

export default function App() {
  return (
    <BrowserRouter basename={basename || undefined}>
      <AuthGate>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<KBListPage />} />
        <Route path="/kb/:kbId" element={<KBLayout />}>
          <Route index element={<Navigate to="documents" replace />} />
          <Route path="documents"  element={<DocumentsPage />} />
          <Route path="search"     element={<SearchTestPage />} />
          <Route path="settings"   element={<KBSettingsPage />} />
          <Route path="comparison" element={<ComparisonPage />} />
          <Route path="audit"      element={<AuditPage />} />
          <Route path="graph"      element={<GraphPage />} />
          <Route path="chat"       element={<ChatPage />} />
        </Route>
      </Routes>
      </AuthGate>
    </BrowserRouter>
  )
}
