# SDR Agent

> 把销售通话录音或转写文本，自动变成可直接入库的 CRM SQL 线索。

支持两种输入：

- **录音文件**（mp3 / wav / m4a …）→ 自动语音转写 → 抽取线索，**无需人工听录音转写**
- **转写文本**（已有 ASR/人工转写结果）→ 直接抽取线索

输出 15 个字段的结构化线索（企业抬头、行业、地区、门店数、设计师人数、客户需求、意向等级等），并对低置信度线索提供人工复核与反馈闭环。

- **企业抬头还原**：从口语简称、品牌名、转写错误还原工商全称
- **意向度识别**：高/中/低三档 + 可回溯的判断依据
- **人工干预台**：低置信度线索审核、修正、批量订正、反馈沉淀

---

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.10+（开发验证于 3.11） |
| 操作系统 | Windows / macOS / Linux |
| 网络 | 可访问所配置的大模型 API |
| 大模型凭证 | 至少一个：DeepSeek / 通义千问（DashScope）/ OpenAI |
| 录音转写凭证 | `DASHSCOPE_API_KEY`（仅使用录音输入时需要） |

联网检索与录音转写都是可选能力，不可用时自动降级或给出明确错误，不影响文本输入的主流程。

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone https://github.com/Ilove117/sdr-agent.git
cd sdr-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

依赖 6 个：`requests`、`httpx`、`pydantic`、`fastapi`、`uvicorn`、`python-multipart`。

### 2. 配置凭证

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`，至少填入一个模型 Key：

```ini
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> `.env` 已加入 `.gitignore`，不会被提交。也可以直接用系统环境变量，程序会优先读取已存在的环境变量。

### 3. 自检（不调用大模型）

```bash
python run.py selfcheck
```

预期输出：`自检结果：全部通过`。这一步验证字段口径、数值抽取、地区/手机号归一、意向评分卡是否正常。

### 4. 跑一遍样例

```bash
python run.py analyze
```

对内置 4 条通话记录逐条分析并打印结果，末尾给出批次汇总：

| 通话 | 企业抬头 | 意向 | 置信度 | 需复核 |
|---|---|---|---|---|
| call-001 | 上海木里实业发展有限公司 | high | 0.653 | 是 |
| call-002 | 恒美居（弃权，候选待人工确认） | medium | 0.584 | 是 |
| call-003 | 未识别（客户未提企业名） | low | 0.351 | 是 |
| call-004 | 德华兔宝宝装饰新材股份有限公司 | high | 0.848 | 否 |

结果同时写入 `data/leads.jsonl`，可直接在工作台「线索库」查看。上表为示例输出，置信度会随模型版本与检索结果小幅浮动，判定结论应保持稳定。

### 5. 启动 Web 工作台

```bash
python run.py serve
```

打开 <http://127.0.0.1:8000>，页面分三个视图：

**工作台** —— 业务用户的主入口

- 拖拽 mp3/wav 录音，或切换到「转写文本」粘贴对话
- 点「开始分析」后显示真实处理阶段与耗时；分析在后台执行，关闭页面不会中断，完成后结果写入线索库
- 结果卡片展示：企业抬头、意向等级、置信度、行业/地区/联系人/手机号/规模、客户需求、判断依据、建议下一步
- 需要人工确认的事项直接列在结果里；可一键复制或下载 CRM JSON
- 「查看并修正」跳转到线索库对应的记录

**线索库** —— 复核人的工作区

- 按复核状态、意向等级、关键词筛选
- 详情 5 个 tab：结构化字段 / 抽取证据 / 抬头还原 / 意向评分 / 复核与反馈
- 抬头还原 tab 里可直接「采用」候选企业；改完点「保存修正」
- 勾选多条可批量通过或批量改意向等级

**设置** —— 系统配置与沉淀

- 当前模型、语音转写、检索通道、复核阈值
- 线索总量、待复核数、意向分布、平均置信度
- 人工订正沉淀的「口述别称 → 工商全称」清单
- 导出已审核线索

---

## 录音输入（无需人工转写）

真实 SDR 通话通常是 mp3 录音。系统内置语音转写，直接吃音频：

```bash
# 先单独看转写质量（不跑线索抽取）
python run.py transcribe --audio data/audio/call_0901.mp3

# 转写并抽取线索，结果写入线索库
python run.py analyze --audio data/audio/call_0901.mp3

# 批量处理一个目录
python run.py analyze --audio-dir data/audio/

# 保留逐句时间戳与说话人信息
python run.py transcribe --audio data/audio/call_0901.mp3 --json
```

打开工作台，切到「录音文件」，拖拽或选择音频；点「开始分析」后转写与抽取一次完成，结果直接出现在右侧。

HTTP 接口：

```bash
# 提交分析任务：立即返回 job_id（202），转写与抽取在后台执行
curl -X POST http://127.0.0.1:8000/api/analyze-audio \
  -F "file=@data/audio/call_0901.mp3" \
  -F "save=true"

# 轮询任务：stages 是真实阶段，succeeded 时 result 为线索
curl http://127.0.0.1:8000/api/jobs/<job_id>
```

## 配置说明

完整配置项见 [.env.example](.env.example)。常用项：

### 大模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 空 | DeepSeek 凭证 |
| `DASHSCOPE_API_KEY` | 空 | 通义千问凭证（同时用于联网检索） |
| `OPENAI_API_KEY` | 空 | OpenAI 凭证（可选） |
| `SDR_PRIMARY_PROVIDER` | `deepseek` | 主模型，取值 `deepseek` / `dashscope` / `openai` |
| `SDR_FALLBACK_PROVIDER` | `dashscope` | 主模型失败时自动降级到的备用模型 |
| `SDR_MODEL_DEEPSEEK` | `deepseek-chat` | 覆盖默认模型名 |
| `SDR_MODEL_DASHSCOPE` | `qwen-plus` | 覆盖默认模型名 |
| `SDR_MODEL_OPENAI` | `gpt-4o-mini` | 覆盖默认模型名 |

### 联网检索（企业抬头还原用）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SDR_ENABLE_WEB_SEARCH` | `1` | 设为 `0` 可完全关闭外部检索 |
| `SDR_SEARCH_PROVIDERS` | `dashscope,html` | 检索通道，按顺序尝试，失败自动跳下一个 |

- `dashscope`：通义千问原生联网检索，质量最好，需要账户可用
- `html`：中文搜索引擎（360 / 搜狗）HTML 检索，无需额外凭证，best-effort

### 语音转写（录音输入用）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SDR_ASR_PROVIDER` | `dashscope` | 转写通道，当前实现 `dashscope` |
| `SDR_ASR_MODEL` | `paraformer-v2` | 支持说话人分离的模型 |
| `SDR_ASR_LANGUAGE` | `zh` | 识别语种 |
| `SDR_ASR_DIARIZATION` | `1` | 是否开启说话人分离（仅单声道生效） |
| `SDR_ASR_POLL_TIMEOUT` | `600` | 轮询转写结果的超时（秒） |

### 人工复核闸门

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SDR_REVIEW_THRESHOLD` | `0.75` | 整体置信度低于此值 → 进入人工复核队列 |
| `SDR_CRITICAL_THRESHOLD` | `0.70` | 关键字段（企业抬头 / 意向等级）低于此值 → 进入复核队列 |

### 服务与稳健性

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SDR_API_HOST` | `127.0.0.1` | Web 工作台监听地址 |
| `SDR_API_PORT` | `8000` | Web 工作台端口 |
| `SDR_REQUEST_TIMEOUT` | `90` | 单次模型请求超时（秒） |
| `SDR_MAX_RETRIES` | `2` | 失败重试次数（429/5xx 自动重试） |
| `SDR_TEMPERATURE` | `0.1` | 模型温度，建议保持低值以保证稳定性 |

---

## 使用方式

### 命令行

```bash
# 分析内置样例
python run.py analyze

# 分析指定文件
python run.py analyze --file data/samples/sample_calls.json

# 只跑某几条，并打印逐字段置信度与候选企业
python run.py analyze --only call-001,call-004 -v

# 直接传入一段通话文本
python run.py analyze --text "销售：您好，是张总吗？……" --call-id cli-001

# 查看线索库统计
python run.py stats
```

输入文件格式（JSON）：

```json
{
  "calls": [
    {
      "id": "call-001",
      "call_time": "2026-10-14 10:32:00",
      "scenario": "可选，仅用于说明",
      "dialogue": "销售：您好……\n客户：……"
    }
  ]
}
```

`dialogue` 建议保留 `说话人：内容` 的换行格式。程序会区分销售与客户话轮，只在客户话轮里提取企业名、地区、规模等事实——避免把「我是酷家乐的小王」当成客户企业。

### 输出

严格输出 CRM SQL 字段：

```json
{
  "company_name": "上海木里实业发展有限公司",
  "industry": "定制家居",
  "area": "浙江省杭州市",
  "contact_name": "张",
  "contact_title": "负责人/总经理",
  "phone": "138****1234",
  "intent_level": "high",
  "intent_reason": "判定为高意向（总分 75.5/100…）；预算明确度 18/24（预算表述：一年几万块）……",
  "demand_summary": "设计部 15 人用现软件出图慢、与工厂对接易出错……",
  "mendian": null,
  "shejishi": 15,
  "call_time": "2026-10-14 10:32:00",
  "next_action": "24 小时内电话回访，发送企业版方案与报价，约演示时间",
  "confidence_score": 0.653,
  "need_human_review": true
}
```

字段口径的唯一来源是 `app/schema.py` 的 `SQL_FIELDS` 与 `app/rules/sql_fields.py`。

### HTTP 接口示例

```bash
# 提交文本分析任务，返回 job_id
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"dialogue":"销售：您好，是张总吗？\n客户：是的，我们做全屋定制的，叫木里木外。","save":true}'

# 轮询任务；succeeded 时 result 就是 CRM 线索
curl http://127.0.0.1:8000/api/jobs/<job_id>

# 修正企业抬头并标记为已修正
curl -X POST http://127.0.0.1:8000/api/leads/call-001/review \
  -H "Content-Type: application/json" \
  -d '{"changes":{"company_name":"杭州木里木外家居有限公司"},"reviewer":"zhangsan"}'

# 批量改意向等级
curl -X POST http://127.0.0.1:8000/api/batch/review \
  -H "Content-Type: application/json" \
  -d '{"ids":["call-001","call-002"],"changes":{"intent_level":"medium"},"reviewer":"zhangsan"}'
```

PowerShell 下把 `curl` 换成 `curl.exe`，或使用 `Invoke-RestMethod`。

---

## 测试与评测

```bash
# 单元测试：76 项，全程离线（注入 FakeLLM / FakeSearch / FakeASR，不消耗额度）
python -m unittest discover -s tests -t .

# 规则层自检
python run.py selfcheck

# golden set 评测（会真实调用模型与检索）
python run.py eval --json
```

评测集在 `data/eval/golden_cases.json`，8 条用例覆盖典型场景：品牌简称、地区冲突、经销商口述、需总部审批、明确拒绝、企业名不可核实等。报告会输出企业抬头准确率、意向等级准确率、自动通过率与平均置信度，并写入 `data/eval_report.json`。

扩充评测集是提升准确率最有效的方式——把真实历史通话脱敏后按同样格式加入即可。

---

## 目录结构

```
run.py                    CLI 入口（analyze / serve / stats / selfcheck / eval）
requirements.txt          依赖
.env.example              配置模板

app/
├── config.py             配置（provider、阈值、路径、ASR）
├── llm.py                模型客户端（多 provider、失败降级、JSON 容错、联网检索）
├── asr.py                语音转写（上传、提交、轮询、说话人角色映射）
├── search.py             检索通道（dashscope / html，自动降级）
├── dialogue.py           话轮解析（区分销售 / 客户）
├── schema.py             输出契约（15 个 CRM 字段）
├── storage.py            线索库与复核状态、反馈日志
├── feedback_memory.py    人工订正 → 别名记忆
├── tracing.py            链路追踪
├── eval.py               评测逻辑
├── rules/                字段口径、行业词库、意向评分卡、抬头规整规则
├── agent/                工作流：抽取 / 抬头还原 / 编排 / 提示词
├── kb/                   企业别名词典（32 条种子数据）
├── api/                  FastAPI 后端（server.py 接口 / jobs.py 后台任务）
└── ui/                   Web 工作台单页（工作台 / 线索库 / 设置）

data/
├── samples/              演示通话记录（文本）
├── eval/                 评测集
├── audio/                运行时生成：上传的录音（已 gitignore）
├── transcripts/          运行时生成：transcribe --json 的转写结果
├── leads.jsonl           运行时生成：线索库
├── feedback.jsonl        运行时生成：人工订正事件
└── traces/               运行时生成：链路追踪

tests/                    离线单元测试
```

`data/leads.jsonl`、`data/feedback.jsonl`、`data/traces/`、`data/audio/`、`data/transcripts/`
已在 `.gitignore` 中，运行时自动创建。

---
