import { useEffect, useRef, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Table, Button, Tag, Space, Typography, Input, Tabs, Drawer, Alert,
  Modal, Form, Select, Upload, Slider, Switch, message, Progress, Tooltip, Popconfirm, Radio,
} from 'antd'
import {
  UploadOutlined, SearchOutlined, DeleteOutlined,
  ReloadOutlined, SettingOutlined,
} from '@ant-design/icons'
import type { UploadFile } from 'antd'
import { docApi, jobApi, kbApi, qaApi } from '../../api'
import dayjs from 'dayjs'
import JobCenter from '../../components/JobCenter'
import DeepCheckView from '../../components/DeepCheckView'
import type { DeepCheckReport, Document, IndexStatus, ProductCapabilities, ProductJob } from '../../types'

const { Title, Text } = Typography

const STATUS_MAP: Record<string, { label: string; color: string }> = {
  uploading: { label: '上传中', color: 'processing' },
  queued:    { label: '排队中', color: 'processing' },
  pending:   { label: '排队中', color: 'processing' },
  parsing:   { label: '解析中', color: 'processing' },
  chunking:  { label: '切块中', color: 'processing' },
  embedding: { label: '向量化', color: 'processing' },
  graphing:  { label: '图谱抽取', color: 'processing' },
  validating:{ label: '校验中', color: 'processing' },
  deleting:  { label: '删除中', color: 'processing' },
  ready:     { label: '就绪', color: 'success' },
  failed:    { label: '失败', color: 'error' },
  timeout:   { label: '超时', color: 'error' },
  cancelled: { label: '已取消', color: 'default' },
  consistency_failed: { label: '校验未通过', color: 'error' },
}

const ACTIVE_STATUS = ['uploading', 'queued', 'pending', 'parsing', 'chunking', 'embedding', 'graphing', 'validating', 'deleting']
const RETRY_STATUS = ['failed', 'timeout', 'consistency_failed']

const STRATEGY_HINT: Record<string, string> = {
  smart: '表格按行，问答按对，制度按条；很短的说明整篇一块。有标题按章节，其余长文按句读切开。',
  delimiter: '只在分隔符处切开。单段超过最大长度时，再按句读切开。',
  fixed: '按最大长度滑动窗口切开，相邻块保留重叠。',
  recursive: '先按空行、换行、句号切开，表格保持整块。',
  ledger: '每一行是一条记录，列名写进文本，不会把一条记录切成两半。',
  qa: '识别“问/问题”和“答/答案”，一对一块。',
  clause: '按“第×条”切开，每条前面带上所属编、章、节。',
  one: '全文作为一块，不再按长度切开。',
}

function formatSize(size: number) {
  if (!size) return '-'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

export default function DocumentsPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [docs, setDocs]     = useState<any[]>([])
  const [total, setTotal]   = useState(0)
  const [page, setPage]     = useState(1)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(false)
  const [importing, setImporting] = useState(false)
  const [form] = Form.useForm()
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [step, setStep] = useState<1 | 2>(1)
  const [chunks, setChunks] = useState<any[]>([])
  const [chunkTotal, setChunkTotal] = useState(0)
  const [chunkPage, setChunkPage] = useState(1)
  const [qaItems, setQaItems] = useState<any[]>([])
  const [qaTotal, setQaTotal] = useState(0)
  const [qaPage, setQaPage] = useState(1)
  const [addingQa, setAddingQa] = useState(false)
  const [uploadingRow, setUploadingRow] = useState<any>(null)
  const [qaForm] = Form.useForm()
  const [limits, setLimits] = useState<ProductCapabilities | null>(null)
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null)
  const [jobsOpen, setJobsOpen] = useState(false)
  const [detail, setDetail] = useState<Document | null>(null)
  const [deleteJob, setDeleteJob] = useState<ProductJob | null>(null)
  const [checkReport, setCheckReport] = useState<DeepCheckReport | null>(null)
  const [rebuildJob, setRebuildJob] = useState<ProductJob | null>(null)
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const chunkStrategy = Form.useWatch('chunk_strategy', form)

  const loadChunks = () => {
    docApi.chunks({ kb_id: kbId, page: chunkPage, page_size: 10, search })
      .then((r: any) => { setChunks(r.items ?? []); setChunkTotal(r.total ?? 0) })
  }
  const loadQa = () => {
    qaApi.list({ kb_id: kbId, page: qaPage, page_size: 10, search })
      .then((r: any) => { setQaItems(r.items ?? []); setQaTotal(r.total ?? 0) })
  }

  const load = () => {
    setLoading(true)
    docApi.list({ kb_id: kbId, page, page_size: 10, search })
      .then((r: any) => { setDocs(r.items ?? []); setTotal(r.total ?? 0) })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [kbId, page, search])
  useEffect(() => {
    kbApi.capabilities().then(setLimits).catch(() => setLimits(null))
    kbApi.indexStatus(kbId).then(setIndexStatus).catch(() => setIndexStatus(null))
  }, [kbId])
  useEffect(() => { loadChunks() }, [kbId, chunkPage, search])
  useEffect(() => { loadQa() }, [kbId, qaPage, search])

  // 轮询处理中文档的状态
  useEffect(() => {
    const processing = docs.filter(d => ACTIVE_STATUS.includes(d.status))
    if (processing.length > 0) {
      pollingRef.current = setInterval(() => load(), 3000)
    }
    return () => { if (pollingRef.current) clearInterval(pollingRef.current) }
  }, [docs])

  const handleUpload = async () => {
    if (step === 1) {
      if (fileList.length === 0) { message.error('请先选择文件'); return }
      setStep(2)
      return
    }
    const vals = await form.validateFields()
    // 兼容 RcFile（直接 File 对象）和 UploadFile（含 originFileObj）两种情况
    const rawFile = (fileList[0] as any)?.originFileObj ?? fileList[0]
    if (!rawFile) { message.error('请选择文件'); return }

    const fd = new FormData()
    fd.append('file', rawFile)
    fd.append('kb_id', kbId)
    fd.append('chunk_strategy', vals.chunk_strategy)
    fd.append('chunk_size', String(vals.chunk_size ?? 512))
    fd.append('chunk_overlap', String(vals.chunk_overlap ?? 64))
    fd.append('chunk_delimiter', vals.chunk_delimiter || '\\n\\n')

    setUploadingRow({
      id: 'local-upload',
      name: rawFile.name,
      chunk_count: 0,
      char_count: 0,
      file_size: rawFile.size || 0,
      updated_at: new Date().toISOString(),
      status: 'uploading',
      progress: 8,
    })
    setImporting(false)
    setFileList([])
    setStep(1)
    form.resetFields()

    try {
      await docApi.upload(fd)
      message.success('文件已上传，正在解析')
      setUploadingRow(null)
      load()
    } catch (e: any) {
      setUploadingRow(null)
      message.error(typeof e === 'string' ? e : '上传失败')
    }
  }

  const columns = [
    {
      title: '名称',
      dataIndex: 'name',
      ellipsis: true,
      render: (name: string, row: Document) => (
        <Button type="link" style={{ padding: 0, height: 'auto', whiteSpace: 'normal', textAlign: 'left' }} onClick={() => setDetail(row)}>
          <Text strong style={{ overflowWrap: 'anywhere' }}>{name}</Text>
        </Button>
      ),
    },
    {
      title: '段落·字符数',
      render: (_: any, r: any) => (
        <Text type="secondary">{r.chunk_count} 段 · {r.char_count?.toLocaleString()} 字</Text>
      ),
    },
    {
      title: '文件大小',
      dataIndex: 'file_size',
      width: 110,
      render: (s: number) => formatSize(s),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      render: (t: string) => dayjs(t).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '状态',
      dataIndex: 'status',
      render: (s: string, r: any) => {
        const info = STATUS_MAP[s] ?? { label: s, color: 'default' }
        if (ACTIVE_STATUS.includes(s)) {
          return (
            <Space>
              <Tag color={info.color}>{info.label}</Tag>
              <Progress percent={r.progress ?? 0} size="small" style={{ width: 80 }} />
            </Space>
          )
        }
        if (r.error_msg && (s === 'failed' || s === 'timeout' || s === 'consistency_failed')) {
          return <Tooltip title={r.error_msg}><Tag color={info.color}>{info.label}</Tag></Tooltip>
        }
        return <Tag color={info.color}>{info.label}</Tag>
      },
    },
    {
      title: '操作',
      render: (_: any, r: any) => (
        <Space>
          <Popconfirm
          title="确认删除该文档？"
          disabled={String(r.id).startsWith('upload-') || r.id === 'local-upload'}
          onConfirm={async () => {
            if (String(r.id).startsWith('upload-') || r.id === 'local-upload') return
            const job = await docApi.delete(r.id, kbId)
            setDeleteJob(job)
            setCheckReport(null)
            message.success('已开始删除')
            load()
          }}
        >
          <Button type="link" danger icon={<DeleteOutlined />} size="small">删除</Button>
        </Popconfirm>
        {ACTIVE_STATUS.includes(r.status) && r.job_id ? (
          <Button
            type="link"
            size="small"
            onClick={async () => {
              await jobApi.cancel(r.job_id)
              message.success('已取消')
              load()
            }}
          >
            取消
          </Button>
        ) : null}
        {RETRY_STATUS.includes(r.status) && r.job_id ? (
          <Button
            type="link"
            size="small"
            onClick={async () => {
              await jobApi.retry(r.job_id)
              message.success('已重新排队')
              load()
            }}
          >
            重试
          </Button>
        ) : null}
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: 16, minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16, gap: 12, flexWrap: 'wrap' }}>
        <div style={{ minWidth: 0 }}>
          <Title level={4} style={{ margin: 0 }}>资料</Title>
          <Text type="secondary">上传个人资料，解析后建立片段和知识关联</Text>
        </div>
        <Space wrap>
          <Input
            prefix={<SearchOutlined />}
            placeholder="搜索文档"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ width: 180, maxWidth: '100%' }}
            allowClear
          />
          <Tooltip title="刷新">
            <Button aria-label="刷新资料" icon={<ReloadOutlined />} onClick={load} />
          </Tooltip>
          <Button data-testid="open-jobs" onClick={() => setJobsOpen(true)}>任务</Button>
          <Button type="primary" icon={<UploadOutlined />} onClick={() => setImporting(true)}>
            导入文件
          </Button>
        </Space>
      </div>
      {indexStatus?.status === 'index_rebuild_required' ? (
        <Alert
          style={{ marginBottom: 12 }}
          type="warning"
          showIcon
          data-testid="index-mismatch"
          message="需要重新构建索引"
          description={indexStatus.reason || '当前模型或切块方式与已有索引不一致。'}
          action={
            <Button
              size="small"
              onClick={async () => {
                const job = await kbApi.rebuildIndex(kbId)
                setRebuildJob(job)
                message.success('已开始重建索引')
              }}
            >
              重建索引
            </Button>
          }
        />
      ) : null}
      {rebuildJob ? (
        <Alert
          style={{ marginBottom: 12 }}
          type={rebuildJob.status === 'failed' ? 'error' : 'info'}
          message={`索引重建 ${rebuildJob.status || '已排队'}`}
          description={rebuildJob.error || rebuildJob.error_message || ''}
          action={
            <Space>
              {['failed', 'timeout'].includes(rebuildJob.status || '') ? (
                <Button size="small" onClick={async () => setRebuildJob(await jobApi.retry(rebuildJob.job_id))}>重试</Button>
              ) : null}
              {['queued', 'running'].includes(rebuildJob.status || '') ? (
                <Button size="small" onClick={async () => setRebuildJob(await jobApi.cancel(rebuildJob.job_id))}>取消</Button>
              ) : null}
            </Space>
          }
        />
      ) : null}

      <Tabs
        items={[
          {
            key: 'file', label: `文件 (${total})`,
            children: (
              <Table
                rowKey="id"
                columns={columns}
                dataSource={uploadingRow && !docs.some(item => item.name === uploadingRow.name) ? [uploadingRow, ...docs] : docs}
                loading={loading}
                pagination={{ total, current: page, pageSize: 10, onChange: setPage }}
                scroll={{ x: 720 }}
                size="middle"
              />
            ),
          },
          {
            key: 'qa', label: `问答 (${qaTotal})`,
            children: (
              <>
                <div style={{ marginBottom: 12, textAlign: 'right' }}>
                  <Button type="primary" onClick={() => setAddingQa(true)}>新建问答</Button>
                </div>
                <Table
                  rowKey="id"
                  dataSource={qaItems}
                  pagination={{ total: qaTotal, current: qaPage, pageSize: 10, onChange: setQaPage }}
                  columns={[
                    { title: '问题', dataIndex: 'question', ellipsis: true },
                    { title: '答案', dataIndex: 'answer', ellipsis: true },
                    { title: '来源', dataIndex: 'source', width: 90, render: (s: string) => s === 'chat' ? '对话' : '手工' },
                    { title: '关联文档', dataIndex: 'doc_name', ellipsis: true, render: (v: string) => v || '-' },
                    { title: '更新时间', dataIndex: 'updated_at', width: 160, render: (t: string) => t ? dayjs(t).format('YYYY-MM-DD HH:mm') : '-' },
                    {
                      title: '操作',
                      width: 80,
                      render: (_: any, r: any) => (
                        <Popconfirm title="删除这条问答？" onConfirm={async () => { await qaApi.delete(r.id, kbId); loadQa() }}>
                          <Button type="link" danger size="small">删除</Button>
                        </Popconfirm>
                      ),
                    },
                  ]}
                />
              </>
            ),
          },
          {
            key: 'chunk', label: `文本块 (${chunkTotal})`,
            children: (
              <Table
                rowKey="chunk_id"
                dataSource={chunks}
                pagination={{ total: chunkTotal, current: chunkPage, pageSize: 10, onChange: setChunkPage }}
                columns={[
                  { title: '文档', dataIndex: 'doc_name', ellipsis: true, width: 220 },
                  { title: '序号', dataIndex: 'index', width: 70 },
                  { title: '字数', dataIndex: 'char_count', width: 80 },
                  { title: '内容', dataIndex: 'content', ellipsis: true },
                ]}
              />
            ),
          },
        ]}
      />

      {/* 导入弹窗 */}
      <Modal
        title={
          <Space>
            <span style={{ color: '#7c5cfc', fontWeight: 600 }}>① 文件上传</span>
            <span style={{ color: '#ccc' }}>——</span>
            <span style={{ color: step === 2 ? '#7c5cfc' : '#ccc', fontWeight: step === 2 ? 600 : 400 }}>② 配置文件</span>
          </Space>
        }
        open={importing}
        onOk={handleUpload}
        onCancel={() => {
          if (step === 2) { setStep(1); return }
          setImporting(false); setFileList([]); form.resetFields()
        }}
        okText={step === 1 ? '下一步' : '完成'}
        cancelText={step === 2 ? '上一步' : '取消'}
        width={600}
      >
        {step === 1 ? (
          <Upload.Dragger
            fileList={fileList}
            beforeUpload={(file) => {
              if (limits && file.size > limits.max_upload_bytes) {
                message.error('文件超过当前允许的大小')
                return Upload.LIST_IGNORE
              }
              setFileList([file as any]); return false
            }}
            onRemove={() => setFileList([])}
            accept={limits?.extensions?.join(',') || undefined}
            maxCount={1}
            style={{ marginTop: 16 }}
          >
            <p className="ant-upload-drag-icon"><UploadOutlined style={{ fontSize: 32, color: '#595959' }} /></p>
            <p className="ant-upload-text">点击或拖拽文件到此处上传</p>
            <p className="ant-upload-hint">
              {limits
                ? `允许 ${limits.extensions.join(' ')}，单个文件不超过 ${(limits.max_upload_bytes / 1024 / 1024).toFixed(0)}MB，每个库最多 ${limits.max_files_per_kb} 个文件`
                : '正在读取上传限制'}
            </p>
          </Upload.Dragger>
        ) : (
          <Form form={form} layout="vertical" style={{ marginTop: 16 }}
            initialValues={{ chunk_strategy: 'smart', chunk_size: 512, chunk_overlap: 64, chunk_delimiter: '\\n\\n' }}
          >
            <Form.Item label="切片策略" name="chunk_strategy" extra={STRATEGY_HINT[chunkStrategy || 'smart']}>
              <Radio.Group>
                <Space direction="vertical">
                  <Radio value="smart">智能切分（推荐，系统自动判断）</Radio>
                  <Radio value="delimiter">分隔符切分（支持自定义分隔符）</Radio>
                  <Radio value="fixed">固定长度切分</Radio>
                  <Radio value="recursive">递归切分（适合长文档）</Radio>
                  <Radio value="ledger">台账切分（Excel / CSV 一行一块）</Radio>
                  <Radio value="qa">问答切分（一问一答一块）</Radio>
                  <Radio value="clause">条款切分（按“第×条”，保留章节名）</Radio>
                  <Radio value="one">整篇一块（适合很短的单页说明）</Radio>
                </Space>
              </Radio.Group>
            </Form.Item>
            {chunkStrategy === 'delimiter' && (
              <Form.Item label="自定义分隔符" name="chunk_delimiter" extra="用 \\n 表示换行，\\t 表示制表符。默认按空行切开。">
                <Input placeholder="\\n\\n" />
              </Form.Item>
            )}
            <Form.Item label="切片最大长度" name="chunk_size">
              <Slider min={64} max={4096} step={64} marks={{ 256: '256', 512: '512', 1024: '1024', 2048: '2048' }} />
            </Form.Item>
            <Form.Item label="重叠长度" name="chunk_overlap">
              <Slider min={0} max={512} step={32} marks={{ 0: '0', 64: '64', 128: '128', 256: '256' }} />
            </Form.Item>
          </Form>
        )}
      </Modal>

      <Modal
        title="新建问答"
        open={addingQa}
        okText="保存"
        onCancel={() => { setAddingQa(false); qaForm.resetFields() }}
        onOk={async () => {
          const vals = await qaForm.validateFields()
          const doc = docs.find(item => item.id === vals.doc_id)
          await qaApi.create({
            kb_id: kbId,
            question: vals.question,
            answer: vals.answer,
            doc_id: vals.doc_id,
            doc_name: doc?.name,
          })
          message.success('问答已保存')
          setAddingQa(false)
          qaForm.resetFields()
          loadQa()
        }}
      >
        <Form form={qaForm} layout="vertical">
          <Form.Item name="question" label="问题" rules={[{ required: true, message: '请输入问题' }]}>
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="answer" label="答案" rules={[{ required: true, message: '请输入答案' }]}>
            <Input.TextArea rows={4} />
          </Form.Item>
          <Form.Item name="doc_id" label="关联文档（可选）">
            <Select allowClear placeholder="选择文档" options={docs.map(item => ({ value: item.id, label: item.name }))} />
          </Form.Item>
        </Form>
      </Modal>
      <JobCenter kbId={kbId} open={jobsOpen} onClose={() => setJobsOpen(false)} />
      <Drawer title="文档详情" open={Boolean(detail)} onClose={() => setDetail(null)} width={Math.min(420, window.innerWidth - 16)}>
        {detail ? (
          <Space direction="vertical" style={{ width: '100%' }} data-testid="doc-detail">
            <Text strong style={{ overflowWrap: 'anywhere' }}>{detail.name}</Text>
            <Text>状态 {detail.status || '—'}</Text>
            <Text>切块 {detail.chunk_count ?? 0}</Text>
            <Text>字符 {detail.char_count ?? 0}</Text>
            {detail.version ? <Text>版本 {detail.version}</Text> : null}
            <Text>创建 {detail.created_at ? dayjs(detail.created_at).format('YYYY-MM-DD HH:mm') : '—'}</Text>
            <Text>更新 {detail.updated_at ? dayjs(detail.updated_at).format('YYYY-MM-DD HH:mm') : '—'}</Text>
            {detail.error_msg ? <Text type="danger" style={{ overflowWrap: 'anywhere' }}>{detail.error_msg}</Text> : null}
          </Space>
        ) : null}
      </Drawer>
      {deleteJob ? (
        <Alert
          style={{ marginTop: 12 }}
          data-testid="delete-job"
          message={`删除任务 ${deleteJob.status || ''}`}
          description={
            <Space direction="vertical">
              <Text>{deleteJob.error || deleteJob.error_message || '删除完成后可以检查是否还有残留。'}</Text>
              {['succeeded', 'failed', 'consistency_failed'].includes(deleteJob.status || '') ? (
                <Space wrap>
                  <Button size="small" onClick={async () => setCheckReport(await kbApi.checkPurge(deleteJob.job_id))}>一致性检查</Button>
                  <Button size="small" data-testid="run-deep-check" onClick={async () => setCheckReport(await kbApi.deepCheck(deleteJob.job_id))}>深度检查</Button>
                </Space>
              ) : null}
              <DeepCheckView report={checkReport} />
            </Space>
          }
        />
      ) : null}
    </div>
  )
}
