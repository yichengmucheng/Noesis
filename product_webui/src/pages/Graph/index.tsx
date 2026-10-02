import { useEffect, useRef, useState, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import {
  Card, Tabs, Table, Tag, Input, Select, Button, Space, Drawer,
  Statistic, Row, Col, Typography, Form, Badge, InputNumber,
  Tooltip, message, Divider, Empty, Spin, Radio,
} from 'antd'
import {
  SearchOutlined, NodeIndexOutlined, ApartmentOutlined,
  SettingOutlined, ReloadOutlined, PlusOutlined, DeleteOutlined,
} from '@ant-design/icons'
import { graphApi } from '../../api'

const { Text, Title } = Typography
const { TextArea } = Input

const TYPE_PALETTE = ['#1677ff', '#13c2c2', '#fa8c16', '#52c41a', '#722ed1', '#eb2f96', '#2f54eb', '#faad14']

function getColor(type: string) {
  const name = String(type || '')
  if (!name) return '#8c8c8c'
  let hash = 0
  for (let i = 0; i < name.length; i += 1) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
  return TYPE_PALETTE[hash % TYPE_PALETTE.length]
}

function relationLabels(text: string) {
  const labels = String(text || '').split(/[；;，,、|]/).map(item => item.trim()).filter(Boolean)
  return labels.length ? labels : ['关联']
}

function WrapText({ text, strong }: { text: string; strong?: boolean }) {
  const value = text || '—'
  return (
    <Tooltip title={value}>
      <div style={{
        whiteSpace: 'normal',
        wordBreak: 'break-word',
        overflowWrap: 'anywhere',
        lineHeight: 1.5,
        fontWeight: strong ? 600 : 400,
      }}>{value}</div>
    </Tooltip>
  )
}

// ─── 力导向图（SVG）─────────────────────────────────────────
interface GraphNode { id: string; label: string; type: string; desc: string; frequency: number; x?: number; y?: number; vx?: number; vy?: number }
interface GraphEdge { from: string; to: string; label: string; desc: string; weight: number }

function ForceGraph({
  nodes, edges, onNodeClick,
}: { nodes: GraphNode[]; edges: GraphEdge[]; onNodeClick: (n: GraphNode) => void }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [positions, setPositions] = useState<Map<string, { x: number; y: number }>>(new Map())
  const [dragging, setDragging] = useState<string | null>(null)
  const animRef = useRef<number>()
  const nodesRef = useRef<GraphNode[]>([])
  const WIDTH = 900
  const HEIGHT = 580

  useEffect(() => {
    if (!nodes.length) return
    const placed = nodes.map((n, i) => ({
      ...n,
      x: WIDTH / 2 + Math.cos((i / nodes.length) * 2 * Math.PI) * 200 + Math.random() * 40,
      y: HEIGHT / 2 + Math.sin((i / nodes.length) * 2 * Math.PI) * 200 + Math.random() * 40,
      vx: 0, vy: 0,
    }))
    nodesRef.current = placed
    simulate()
    return () => cancelAnimationFrame(animRef.current!)
  }, [nodes.length, edges.length])

  const simulate = () => {
    const run = () => {
      const ns = nodesRef.current
      const edgeMap = new Map<string, string[]>()
      edges.forEach(e => {
        if (!edgeMap.has(e.from)) edgeMap.set(e.from, [])
        edgeMap.get(e.from)!.push(e.to)
      })
      for (let iter = 0; iter < 3; iter++) {
        for (let i = 0; i < ns.length; i++) {
          ns[i].vx = 0; ns[i].vy = 0
          for (let j = 0; j < ns.length; j++) {
            if (i === j) continue
            const dx = (ns[i].x! - ns[j].x!) || 0.1
            const dy = (ns[i].y! - ns[j].y!) || 0.1
            const d = Math.sqrt(dx * dx + dy * dy) || 1
            const repulse = 2400 / (d * d)
            ns[i].vx! += (dx / d) * repulse
            ns[i].vy! += (dy / d) * repulse
          }
          const links = edgeMap.get(ns[i].id) || []
          links.forEach(tid => {
            const t = ns.find(n => n.id === tid)
            if (!t) return
            const dx = t.x! - ns[i].x!
            const dy = t.y! - ns[i].y!
            const d = Math.sqrt(dx * dx + dy * dy) || 1
            const spring = (d - 140) * 0.04
            ns[i].vx! += dx / d * spring
            ns[i].vy! += dy / d * spring
          })
          const cx = ns[i].x! - WIDTH / 2
          const cy = ns[i].y! - HEIGHT / 2
          ns[i].vx! -= cx * 0.003
          ns[i].vy! -= cy * 0.003
          ns[i].x = Math.max(30, Math.min(WIDTH - 30, ns[i].x! + ns[i].vx! * 0.5))
          ns[i].y = Math.max(30, Math.min(HEIGHT - 30, ns[i].y! + ns[i].vy! * 0.5))
        }
      }
      const pos = new Map<string, { x: number; y: number }>()
      ns.forEach(n => pos.set(n.id, { x: n.x!, y: n.y! }))
      setPositions(new Map(pos))
      animRef.current = requestAnimationFrame(run)
    }
    animRef.current = requestAnimationFrame(run)
  }

  const getPos = (id: string) => positions.get(id) || { x: WIDTH / 2, y: HEIGHT / 2 }
  const getR = (n: GraphNode) => Math.min(30, 10 + (n.frequency || 1) * 2)

  return (
    <svg ref={svgRef} width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      style={{ background: '#0a0e1a', borderRadius: 12, border: '1px solid #1d2d44' }}>
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#4a5568" />
        </marker>
      </defs>
      {/* 边 */}
      {edges.map((e, i) => {
        const s = getPos(e.from); const t = getPos(e.to)
        const dx = t.x - s.x; const dy = t.y - s.y
        const d = Math.sqrt(dx * dx + dy * dy) || 1
        const tr = getR(nodes.find(n => n.id === e.to) || nodes[0])
        const ex = t.x - dx / d * tr; const ey = t.y - dy / d * tr
        const mx = (s.x + ex) / 2; const my = (s.y + ey) / 2
        const strokeW = Math.min(4, 1 + (e.weight || 1) * 0.5)
        return (
          <g key={i}>
            <line x1={s.x} y1={s.y} x2={ex} y2={ey}
              stroke="#2a4a6b" strokeWidth={strokeW}
              markerEnd="url(#arrow)" opacity={0.7} />
            <text x={mx} y={my} fill="#6b8fa8" fontSize={9} textAnchor="middle"
              style={{ userSelect: 'none', pointerEvents: 'none' }}>
              {e.label?.slice(0, 12)}
            </text>
          </g>
        )
      })}
      {/* 节点 */}
      {nodes.map(n => {
        const { x, y } = getPos(n.id)
        const r = getR(n)
        const color = getColor(n.type)
        return (
          <g key={n.id} style={{ cursor: 'pointer' }} onClick={() => onNodeClick(n)}>
            <circle cx={x} cy={y} r={r} fill={color} fillOpacity={0.85}
              stroke={color} strokeWidth={2}
              style={{ filter: 'drop-shadow(0 0 6px currentColor)' }} />
            <text x={x} y={y + r + 12} fill="#c8d8e8" fontSize={10} textAnchor="middle"
              style={{ userSelect: 'none', pointerEvents: 'none' }}>
              {n.label?.slice(0, 8)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ─── 主页面 ──────────────────────────────────────────────────
export default function GraphPage() {
  const { kbId } = useParams<{ kbId: string }>()
  const [activeTab, setActiveTab] = useState('browser')

  // 统计
  const [stats, setStats] = useState<any>(null)
  // 图谱浏览
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([])
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([])
  const [graphLoading, setGraphLoading] = useState(false)
  const [searchEntity, setSearchEntity] = useState('')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [nodeNeighbors, setNodeNeighbors] = useState<any[]>([])
  // 实体
  const [entities, setEntities] = useState<any[]>([])
  const [entityTotal, setEntityTotal] = useState(0)
  const [entityLoading, setEntityLoading] = useState(false)
  const [entityFilter, setEntityFilter] = useState<{ type?: string; search?: string; page: number; sort?: string }>({ page: 1 })
  const [graphQuery, setGraphQuery] = useState<{ name?: string; nonce: number }>({ nonce: 0 })
  const graphReq = useRef(0)
  // 关系
  const [relations, setRelations] = useState<any[]>([])
  const [relTotal, setRelTotal] = useState(0)
  const [relTypeOptions, setRelTypeOptions] = useState<string[]>([])
  const [relLoading, setRelLoading] = useState(false)
  const [relFilter, setRelFilter] = useState<{ entity?: string; relType?: string; page: number }>({ page: 1 })
  // 配置
  const [graphConfig, setGraphConfig] = useState<any>(null)
  const [configLoading, setConfigLoading] = useState(false)
  const [configSaving, setConfigSaving] = useState(false)
  const [configForm] = Form.useForm()

  const loadStats = useCallback(async () => {
    if (!kbId) return
    try {
      const data = await graphApi.getStats(kbId)
      setStats(data)
    } catch { /* 静默 */ }
  }, [kbId])

  const loadSubgraph = useCallback(async (entity?: string) => {
    if (!kbId) return
    const req = ++graphReq.current
    setGraphLoading(true)
    try {
      const data = await graphApi.getSubgraph(kbId, entity || undefined, 2, 80)
      if (req !== graphReq.current) return
      setGraphNodes(data.nodes || [])
      setGraphEdges(data.edges || [])
    } catch (e) {
      if (req === graphReq.current) message.error('图谱加载失败')
    } finally {
      if (req === graphReq.current) setGraphLoading(false)
    }
  }, [kbId])

  const requestGraph = (name?: string) => {
    const next = name?.trim() || undefined
    setSearchEntity(next || '')
    setGraphQuery(current => ({ name: next, nonce: current.nonce + 1 }))
    setActiveTab('browser')
  }

  const loadEntities = useCallback(async () => {
    if (!kbId) return
    setEntityLoading(true)
    try {
      const data = await graphApi.getEntities(kbId, {
        entity_type: entityFilter.type,
        search: entityFilter.search,
        page: entityFilter.page,
        page_size: 20,
        sort: entityFilter.sort || 'frequency_desc',
      })
      setEntities(data.items || [])
      setEntityTotal(data.total || 0)
    } catch { message.error('实体加载失败') }
    finally { setEntityLoading(false) }
  }, [kbId, entityFilter])

  const loadRelations = useCallback(async () => {
    if (!kbId) return
    setRelLoading(true)
    try {
      const data = await graphApi.getRelations(kbId, {
        entity_name: relFilter.entity,
        rel_type: relFilter.relType,
        page: relFilter.page,
        page_size: 20,
      })
      setRelations(data.items || [])
      setRelTotal(data.total || 0)
      if (Array.isArray(data.relation_types)) setRelTypeOptions(data.relation_types)
    } catch { message.error('关系加载失败') }
    finally { setRelLoading(false) }
  }, [kbId, relFilter])

  const loadConfig = useCallback(async () => {
    if (!kbId) return
    setConfigLoading(true)
    try {
      const data = await graphApi.getConfig(kbId)
      setGraphConfig(data)
      configForm.setFieldsValue({
        extraction_mode: data.extraction_mode,
        domain: data.domain,
        gleaning_rounds: data.gleaning_rounds,
        entity_types: (data.entity_types || []).join('\n'),
        relation_types: (data.relation_types || []).join('\n'),
        custom_rules: data.custom_rules,
      })
    } catch { /* 静默 */ }
    finally { setConfigLoading(false) }
  }, [kbId, configForm])

  useEffect(() => { loadStats() }, [loadStats])
  useEffect(() => {
    if (activeTab !== 'browser') return
    loadSubgraph(graphQuery.name)
  }, [activeTab, graphQuery, loadSubgraph])
  useEffect(() => { if (activeTab === 'entities') loadEntities() }, [activeTab, loadEntities])
  useEffect(() => { if (activeTab === 'relations') loadRelations() }, [activeTab, loadRelations])
  useEffect(() => { if (activeTab === 'config') loadConfig() }, [activeTab, loadConfig])

  const handleNodeClick = async (node: GraphNode) => {
    setSelectedNode(node)
    if (!kbId) return
    try {
      const data = await graphApi.getEntityNeighbors(kbId, node.label)
      setNodeNeighbors(data.neighbors || [])
    } catch { setNodeNeighbors([]) }
  }

  const saveConfig = async (values: any) => {
    if (!kbId) return
    setConfigSaving(true)
    try {
      const payload = {
        entity_types:    (values.entity_types || '').split('\n').map((s: string) => s.trim()).filter(Boolean),
        relation_types:  (values.relation_types || '').split('\n').map((s: string) => s.trim()).filter(Boolean),
        extraction_mode: values.extraction_mode,
        domain:          values.domain,
        gleaning_rounds: values.gleaning_rounds,
        custom_rules:    values.custom_rules || '',
        few_shot_examples: '',
      }
      await graphApi.updateConfig(kbId, payload)
      message.success('图谱配置已保存')
      loadConfig()
    } catch (e: any) {
      message.error('保存失败: ' + e)
    } finally {
      setConfigSaving(false)
    }
  }

  // 实体类型分布（legend）
  const typeDist = stats?.entity_type_distribution || []

  const entityColumns = [
    {
      title: '实体名称', dataIndex: 'name', key: 'name',
      render: (v: string) => <Text strong>{v}</Text>,
    },
    {
      title: '类型', dataIndex: 'entity_type', key: 'entity_type',
      render: (v: string) => (
        <Tag color={getColor(v)} style={{ color: '#fff', borderColor: getColor(v) }}>{v}</Tag>
      ),
    },
    {
      title: '描述', dataIndex: 'description', key: 'description',
      ellipsis: true,
      render: (v: string) => <Text type="secondary" ellipsis={{ tooltip: v }}>{v}</Text>,
    },
    {
      title: '出现频次', dataIndex: 'frequency', key: 'frequency', width: 110,
      sorter: true,
      sortOrder: (entityFilter.sort || 'frequency_desc') === 'frequency_asc' ? 'ascend' : 'descend',
      render: (v: number) => <Badge count={v} showZero overflowCount={9999} color="#1677ff" />,
    },
    {
      title: '关联文档数', dataIndex: 'doc_count', key: 'doc_count', width: 100,
      render: (v: number) => <Text>{v}</Text>,
    },
    {
      title: '操作', key: 'action', width: 100,
      render: (_: any, row: any) => (
        <Button size="small" type="link"
          onClick={() => requestGraph(row.name)}>
          查看图谱
        </Button>
      ),
    },
  ]

  const relCell = () => ({
    style: { maxWidth: 0, whiteSpace: 'normal' as const, verticalAlign: 'top' as const, wordBreak: 'break-word' as const },
  })
  const relColumns = [
    {
      title: '源实体', dataIndex: 'source', key: 'source', width: '18%',
      onCell: relCell,
      render: (v: string) => <WrapText text={v} strong />,
    },
    {
      title: '关系类型', dataIndex: 'relation_type', key: 'relation_type', width: '16%',
      onCell: relCell,
      render: (v: string) => (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, minWidth: 0 }}>
          {relationLabels(v).map((label, index) => (
            <Tag key={`${label}-${index}`} color="purple" style={{ margin: 0, whiteSpace: 'normal', height: 'auto', wordBreak: 'break-word' }}>
              {label}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '目标实体', dataIndex: 'target', key: 'target', width: '22%',
      onCell: relCell,
      render: (v: string) => <WrapText text={v} strong />,
    },
    {
      title: '关键词', dataIndex: 'keywords', key: 'keywords', width: '14%',
      onCell: relCell,
      render: (v: string) => <WrapText text={v} />,
    },
    {
      title: '关系描述', dataIndex: 'description', key: 'description', width: '22%',
      onCell: relCell,
      render: (v: string) => <Text type="secondary" style={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>{v || '—'}</Text>,
    },
    {
      title: '强度', dataIndex: 'weight', key: 'weight', width: '8%',
      render: (v: number) => <Text>{(v || 1).toFixed(1)}</Text>,
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 20 }}>
        {[
          { title: '实体总数', value: stats?.entity_count ?? '—', color: '#1677ff' },
          { title: '关系总数', value: stats?.relation_count ?? '—', color: '#52c41a' },
          { title: '关联文档', value: stats?.document_count ?? '—', color: '#faad14' },
          { title: '文本块数', value: stats?.chunk_count ?? '—', color: '#722ed1' },
        ].map(s => (
          <Col span={6} key={s.title}>
            <Card size="small" style={{ borderColor: s.color + '40', background: s.color + '10' }}>
              <Statistic title={s.title} value={s.value}
                valueStyle={{ color: s.color, fontWeight: 700 }} />
            </Card>
          </Col>
        ))}
      </Row>

      <Card bodyStyle={{ padding: 0 }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          style={{ padding: '0 16px' }}
          items={[
            {
              key: 'browser',
              label: <span><NodeIndexOutlined /> 图谱浏览</span>,
              children: (
                <div style={{ padding: '0 16px 16px' }}>
                  <Space style={{ marginBottom: 12 }}>
                    <Input.Search
                      placeholder="搜索实体名称…"
                      value={searchEntity}
                      onChange={e => setSearchEntity(e.target.value)}
                      onSearch={v => requestGraph(v)}
                      style={{ width: 240 }}
                    />
                    <Button icon={<ReloadOutlined />} onClick={() => requestGraph()}>
                      显示全图
                    </Button>
                    <Space wrap>
                      {typeDist.map((item: { type?: string }) => {
                        const typeName = String(item.type || '').trim()
                        if (!typeName) return null
                        return (
                          <Tag
                            key={typeName}
                            color={getColor(typeName)}
                            style={{ color: '#fff', cursor: 'pointer' }}
                            onClick={() => { setEntityFilter(f => ({ ...f, type: typeName, page: 1 })); setActiveTab('entities') }}
                          >
                            {typeName}
                          </Tag>
                        )
                      })}
                    </Space>
                  </Space>

                  <Spin spinning={graphLoading}>
                    {graphNodes.length > 0 ? (
                      <>
                        <div className="graph-canvas" style={{ width: '100%', overflow: 'hidden' }}>
                          <ForceGraph nodes={graphNodes} edges={graphEdges} onNodeClick={handleNodeClick} />
                        </div>
                        <div className="graph-mobile-fallback">
                          {graphNodes.map(node => (
                            <div key={node.id} style={{ padding: '8px 0', borderBottom: '1px solid #f0f0f0', overflowWrap: 'anywhere' }}>
                              <Text strong>{node.label}</Text>
                              <Text type="secondary"> · {node.type}</Text>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <Empty description="还没有知识关联，上传资料后会在这里显示" style={{ padding: 80 }} />
                    )}
                  </Spin>

                  {/* 节点详情抽屉 */}
                  <Drawer
                    title={selectedNode ? `${selectedNode.label} · ${selectedNode.type}` : ''}
                    open={!!selectedNode}
                    onClose={() => setSelectedNode(null)}
                    width={440}
                  >
                    {selectedNode && (
                      <>
                        <Tag color={getColor(selectedNode.type)} style={{ color: '#fff', marginBottom: 12 }}>
                          {selectedNode.type}
                        </Tag>
                        <Title level={5}>{selectedNode.label}</Title>
                        <Text type="secondary">{selectedNode.desc}</Text>
                        <Divider>关联实体（{nodeNeighbors.length}）</Divider>
                        {nodeNeighbors.map((nb, i) => (
                          <div key={i} style={{ marginBottom: 8, padding: '6px 10px',
                            background: '#f5f5f5', borderRadius: 6 }}>
                            <Space>
                              <Tag color={getColor(nb.neighbor_type)} style={{ color: '#fff' }}>
                                {nb.neighbor_type}
                              </Tag>
                              <Text strong style={{ cursor: 'pointer', color: '#1677ff' }}
                                onClick={() => { requestGraph(nb.neighbor_name); setSelectedNode(null) }}>
                                {nb.neighbor_name}
                              </Text>
                              <Tag color="#722ed1" style={{ color: '#fff', fontSize: 11 }}>
                                {nb.relation_kw || nb.relation_type}
                              </Tag>
                            </Space>
                            {nb.description && (
                              <div style={{ marginTop: 4 }}>
                                <Text type="secondary" style={{ fontSize: 12 }}>{nb.description}</Text>
                              </div>
                            )}
                          </div>
                        ))}
                      </>
                    )}
                  </Drawer>
                </div>
              ),
            },
            {
              key: 'entities',
              label: <span><ApartmentOutlined /> 实体管理</span>,
              children: (
                <div style={{ padding: '0 16px 16px' }}>
                  <Space style={{ marginBottom: 12 }}>
                    <Input
                      placeholder="搜索实体名称或描述"
                      prefix={<SearchOutlined />}
                      style={{ width: 240 }}
                      onChange={e => setEntityFilter(f => ({ ...f, search: e.target.value, page: 1 }))}
                    />
                    <Select
                      placeholder="实体类型"
                      allowClear
                      style={{ width: 140 }}
                      onChange={v => setEntityFilter(f => ({ ...f, type: v, page: 1 }))}
                      options={typeDist.map((item: { type?: string }) => ({
                        label: item.type,
                        value: item.type,
                      })).filter((item: { value?: string }) => item.value)}
                    />
                    <Button icon={<ReloadOutlined />} onClick={loadEntities}>刷新</Button>
                  </Space>
                  <Table
                    columns={entityColumns}
                    dataSource={entities}
                    rowKey="name"
                    loading={entityLoading}
                    size="small"
                    onChange={(pagination, _filters, sorter) => {
                      const current = Array.isArray(sorter) ? sorter[0] : sorter
                      const nextSort = current?.order === 'ascend' ? 'frequency_asc' : 'frequency_desc'
                      const prevSort = entityFilter.sort || 'frequency_desc'
                      setEntityFilter(f => ({
                        ...f,
                        sort: nextSort,
                        page: nextSort !== prevSort ? 1 : (pagination.current || 1),
                      }))
                    }}
                    pagination={{
                      total: entityTotal,
                      pageSize: 20,
                      current: entityFilter.page,
                      showTotal: total => `共 ${total} 条`,
                    }}
                  />
                </div>
              ),
            },
            {
              key: 'relations',
              label: <span>关系管理</span>,
              children: (
                <div style={{ padding: '0 16px 16px' }}>
                  <style>{`
                    .kb-rel-table .ant-table { table-layout: fixed; width: 100%; }
                    .kb-rel-table .ant-table-cell {
                      white-space: normal !important;
                      word-break: break-word;
                      overflow-wrap: anywhere;
                      vertical-align: top;
                    }
                  `}</style>
                  <Space style={{ marginBottom: 12 }} wrap>
                    <Input
                      placeholder="源实体或目标实体"
                      prefix={<SearchOutlined />}
                      allowClear
                      style={{ width: 220 }}
                      value={relFilter.entity}
                      onChange={e => setRelFilter(f => ({ ...f, entity: e.target.value, page: 1 }))}
                    />
                    <Select
                      placeholder="关系类型"
                      allowClear
                      showSearch
                      style={{ width: 180 }}
                      value={relFilter.relType}
                      onChange={v => setRelFilter(f => ({ ...f, relType: v, page: 1 }))}
                      options={relTypeOptions.map(t => ({
                        label: t, value: t,
                      }))}
                    />
                    <Button icon={<ReloadOutlined />} onClick={loadRelations}>刷新</Button>
                  </Space>
                  <Table
                    className="kb-rel-table"
                    columns={relColumns}
                    dataSource={relations}
                    rowKey={(r, i) => `${r.source}-${r.relation_type}-${r.target}-${i}`}
                    loading={relLoading}
                    size="small"
                    tableLayout="fixed"
                    pagination={{
                      total: relTotal,
                      pageSize: 20,
                      current: relFilter.page,
                      showTotal: total => `共 ${total} 条`,
                      onChange: p => setRelFilter(f => ({ ...f, page: p })),
                    }}
                  />
                </div>
              ),
            },
            {
              key: 'config',
              label: <span><SettingOutlined /> 图谱配置</span>,
              children: (
                <div style={{ padding: '0 24px 24px', maxWidth: 820 }}>
                  <Spin spinning={configLoading}>
                    <Form form={configForm} layout="vertical" onFinish={saveConfig}>
                      <Row gutter={24}>
                        <Col span={12}>
                          <Form.Item label="抽取模式" name="extraction_mode">
                            <Radio.Group>
                              <Radio.Button value="open">开放三元组</Radio.Button>
                              <Radio.Button value="preset">领域预设</Radio.Button>
                              <Radio.Button value="custom">完全自定义</Radio.Button>
                              <Radio.Button value="auto">自动推断</Radio.Button>
                            </Radio.Group>
                          </Form.Item>
                        </Col>
                        <Col span={6}>
                          <Form.Item label="预设领域" name="domain">
                            <Select options={[
                              { label: '通用知识', value: 'general' },
                              { label: '金融', value: 'finance' },
                              { label: '法律', value: 'law' },
                              { label: '医疗', value: 'medical' },
                              { label: '通用', value: 'general' },
                            ]} />
                          </Form.Item>
                        </Col>
                        <Col span={6}>
                          <Form.Item label="Gleaning 轮数" name="gleaning_rounds">
                            <InputNumber min={0} max={3} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                      </Row>

                      <Row gutter={24}>
                        <Col span={12}>
                          <Form.Item
                            label="实体类型（每行一个）"
                            name="entity_types"
                            help="支持自定义，每行一个实体类型名称">
                            <TextArea rows={10} style={{ fontFamily: 'monospace', fontSize: 13 }}
                              placeholder="公司&#10;基金产品&#10;债券&#10;风险因子&#10;监管机构" />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item
                            label="关系类型（每行一个）"
                            name="relation_types"
                            help="支持自定义，每行一个关系类型">
                            <TextArea rows={10} style={{ fontFamily: 'monospace', fontSize: 13 }}
                              placeholder="投资/持有&#10;监管/被监管&#10;竞争&#10;合作&#10;发行" />
                          </Form.Item>
                        </Col>
                      </Row>

                      <Form.Item label="自定义补充规则" name="custom_rules">
                        <TextArea rows={3} placeholder="可在此输入额外的抽取规则说明（补充到 System Prompt）" />
                      </Form.Item>

                      {graphConfig?.prompt_preview && (
                        <>
                          <Divider>当前 Prompt 预览</Divider>
                          <div style={{ background: '#001529', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                            <Text style={{ color: '#52c41a', fontSize: 12, fontFamily: 'monospace',
                              whiteSpace: 'pre-wrap', display: 'block' }}>
                              {graphConfig.prompt_preview.system}
                            </Text>
                          </div>
                        </>
                      )}

                      <Form.Item>
                        <Space>
                          <Button type="primary" htmlType="submit" loading={configSaving}>
                            保存配置
                          </Button>
                          <Button onClick={loadConfig}>重置</Button>
                        </Space>
                      </Form.Item>
                    </Form>
                  </Spin>
                </div>
              ),
            },
          ]}
        />
      </Card>
    </div>
  )
}
