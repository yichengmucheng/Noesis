import { useEffect, useState } from 'react'
import { Button, Drawer, List, Space, Tag, Typography, message } from 'antd'
import dayjs from 'dayjs'
import { jobApi, kbApi } from '../api'
import { STAGE_LABEL } from '../constants'
import type { ProductJob } from '../types'

const { Text, Paragraph } = Typography

const PIPELINE = ['uploading', 'parsing', 'chunking', 'embedding', 'graphing', 'validating']

export default function JobCenter({
  kbId,
  open,
  onClose,
}: {
  kbId: string
  open: boolean
  onClose: () => void
}) {
  const [items, setItems] = useState<ProductJob[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)

  const load = (next = page) => {
    setLoading(true)
    kbApi.jobs(kbId, { page: next, page_size: 8 })
      .then((result) => {
        setItems(result.items || [])
        setTotal(result.total || 0)
      })
      .catch(() => {
        setItems([])
        setTotal(0)
      })
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    if (open) load(page)
  }, [open, page, kbId])

  return (
    <Drawer
      title="任务"
      open={open}
      onClose={onClose}
      width={Math.min(480, window.innerWidth - 16)}
      data-testid="job-center"
    >
      <List
        loading={loading}
        dataSource={items}
        locale={{ emptyText: '还没有处理任务' }}
        pagination={{
          current: page,
          pageSize: 8,
          total,
          onChange: setPage,
          size: 'small',
        }}
        renderItem={(job) => {
          const failed = ['failed', 'timeout', 'consistency_failed'].includes(job.status || '')
          const active = ['queued', 'running', 'pending', 'parsing', 'chunking', 'embedding', 'graphing', 'validating'].includes(job.status || '')
          return (
            <List.Item data-testid="job-row">
              <Space direction="vertical" style={{ width: '100%' }} size={8}>
                <Space wrap>
                  <Text strong style={{ overflowWrap: 'anywhere' }}>{job.file_name || job.job_type || '任务'}</Text>
                  <Tag>{STAGE_LABEL[job.status || ''] || STAGE_LABEL[job.stage || ''] || job.status || '未知'}</Tag>
                </Space>
                <div style={{ overflowWrap: 'anywhere' }}>
                  {PIPELINE.map((key) => STAGE_LABEL[key]).join(' → ')}
                  {job.stage ? ` · 当前 ${STAGE_LABEL[job.stage] || job.stage}` : ''}
                </div>
                {failed && (job.error || job.error_message) ? (
                  <Paragraph type="danger" style={{ marginBottom: 0, overflowWrap: 'anywhere' }}>
                    {job.error || job.error_message}
                  </Paragraph>
                ) : null}
                <Text type="secondary">
                  {job.created_at ? dayjs(job.created_at).format('YYYY-MM-DD HH:mm') : ''}
                  {job.finished_at ? ` · 完成 ${dayjs(job.finished_at).format('HH:mm')}` : ''}
                </Text>
                <Space wrap>
                  {active ? (
                    <Button
                      size="small"
                      aria-label="取消任务"
                      onClick={async () => {
                        await jobApi.cancel(job.job_id)
                        message.success('已请求取消')
                        load()
                      }}
                    >
                      取消
                    </Button>
                  ) : null}
                  {failed ? (
                    <Button
                      size="small"
                      aria-label="重试任务"
                      onClick={async () => {
                        await jobApi.retry(job.job_id)
                        message.success('已重新排队')
                        load()
                      }}
                    >
                      重试
                    </Button>
                  ) : null}
                </Space>
              </Space>
            </List.Item>
          )
        }}
      />
    </Drawer>
  )
}
