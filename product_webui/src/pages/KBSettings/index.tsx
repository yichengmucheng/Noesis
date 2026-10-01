import { useEffect, useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Typography, Radio, Slider, Switch, Button, Space,
  Form, Select, Tabs, message, Spin,
} from 'antd'
import { kbApi } from '../../api'
import { RECOMMENDED_RETRIEVAL, RETRIEVAL_MODE_OPTIONS } from '../../constants'
import type { KbSettings } from '../../types'

const { Title, Text } = Typography

export default function KBSettingsPage() {
  const { kbId } = useOutletContext<{ kbId: string }>()
  const [settings, setSettings] = useState<KbSettings | null>(null)
  const [saving, setSaving] = useState(false)
  const [form] = Form.useForm()
  const retrievalMode = Form.useWatch('retrieval_mode', form)

  useEffect(() => {
    kbApi.getSettings(kbId).then((value) => {
      setSettings(value)
      form.setFieldsValue({
        retrieval_mode: value.retrieval_mode ?? 'mix',
        top_k: value.top_k ?? 10,
        similarity_ratio: value.similarity_ratio ?? 0.5,
        similarity_threshold: value.similarity_threshold ?? 0.2,
        rerank_enabled: value.rerank_enabled ?? false,
        graph_enabled: value.graph_enabled ?? true,
        context_expand: value.context_expand ?? false,
        answer_detail: value.answer_detail === 'brief' ? 'concise' : value.answer_detail === 'normal' || !value.answer_detail ? 'standard' : value.answer_detail,
        show_citations: value.show_citations !== false,
        ignore_whitespace: value.ignore_whitespace ?? true,
        only_changes: value.only_changes ?? true,
        mark_number_changes: value.mark_number_changes ?? true,
        audit_auto_pass: value.audit_auto_pass ?? false,
        audit_reject_comment: value.audit_reject_comment ?? true,
      })
    })
  }, [kbId])

  const save = async (patch?: Partial<KbSettings>) => {
    const vals = patch || await form.validateFields()
    if (patch) form.setFieldsValue(patch)
    setSaving(true)
    try {
      const saved = await kbApi.updateSettings(kbId, { ...settings, ...form.getFieldsValue(), ...vals })
      setSettings(saved)
      message.success('设置已保存')
    } finally {
      setSaving(false)
    }
  }

  if (!settings) return <div style={{ padding: 48, textAlign: 'center' }}><Spin /></div>

  return (
    <div style={{ padding: 16, minWidth: 0 }}>
      <Title level={4} style={{ marginBottom: 16 }}>设置</Title>
      <Tabs
        items={[
          {
            key: 'basic',
            label: '回答偏好',
            children: (
              <Form form={form} layout="vertical" style={{ maxWidth: 560 }}>
                <Form.Item name="graph_enabled" label="使用知识关联" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Form.Item name="context_expand" label="扩展上下文" valuePropName="checked" extra="打开后回答可以带上命中段落周围的内容。">
                  <Switch />
                </Form.Item>
                <Form.Item name="answer_detail" label="回答详细程度">
                  <Radio.Group>
                    <Radio.Button value="concise">简洁</Radio.Button>
                    <Radio.Button value="standard">标准</Radio.Button>
                    <Radio.Button value="detailed">详细</Radio.Button>
                  </Radio.Group>
                </Form.Item>
                <Form.Item name="show_citations" label="默认显示引用" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Button type="primary" onClick={() => save()} loading={saving}>保存设置</Button>
              </Form>
            ),
          },
          {
            key: 'advanced',
            label: '高级检索设置',
            children: (
              <Form form={form} layout="vertical" style={{ maxWidth: 640 }}>
                <Form.Item name="retrieval_mode" label="检索模式">
                  <Select options={RETRIEVAL_MODE_OPTIONS.map(({ value, label }) => ({ value, label }))} />
                </Form.Item>
                <Form.Item name="top_k" label="返回数量">
                  <Radio.Group>
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(n => (
                      <Radio.Button key={n} value={n}>{n}</Radio.Button>
                    ))}
                  </Radio.Group>
                </Form.Item>
                <Form.Item name="similarity_ratio" label="混合比例">
                  <Slider min={0} max={1} step={0.1} disabled={retrievalMode !== 'hybrid'} />
                </Form.Item>
                <Form.Item name="similarity_threshold" label="相关度阈值">
                  <Slider min={0} max={1} step={0.05} />
                </Form.Item>
                <Form.Item name="rerank_enabled" label="交叉编码重排" valuePropName="checked" extra="模型名称来自当前运行配置，这里不写死。">
                  <Switch />
                </Form.Item>
                <Space wrap>
                  <Button type="primary" onClick={() => save()} loading={saving}>保存设置</Button>
                  <Button onClick={() => save(RECOMMENDED_RETRIEVAL)}>恢复推荐值</Button>
                </Space>
              </Form>
            ),
          },
          {
            key: 'comparison',
            label: '知识比对设置',
            children: (
              <Form form={form} layout="vertical" style={{ maxWidth: 560 }}>
                <Form.Item name="ignore_whitespace" label="忽略空白差异" valuePropName="checked"><Switch /></Form.Item>
                <Form.Item name="only_changes" label="只保留有变化的内容" valuePropName="checked"><Switch /></Form.Item>
                <Form.Item name="mark_number_changes" label="数字变化标为高严重度" valuePropName="checked"><Switch /></Form.Item>
                <Button type="primary" onClick={() => save()} loading={saving}>保存设置</Button>
              </Form>
            ),
          },
          {
            key: 'audit',
            label: '审核设置',
            children: (
              <Form form={form} layout="vertical" style={{ maxWidth: 560 }}>
                <Form.Item name="audit_auto_pass" label="新内容默认视为已通过" valuePropName="checked"><Switch /></Form.Item>
                <Form.Item name="audit_reject_comment" label="拒绝时必须填写原因" valuePropName="checked"><Switch /></Form.Item>
                <Button type="primary" onClick={() => save()} loading={saving}>保存设置</Button>
              </Form>
            ),
          },
        ]}
      />
    </div>
  )
}
