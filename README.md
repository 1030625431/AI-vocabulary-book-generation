# 视觉词汇书生成器

> 给任意一份**英语单词列表**，用 LLM + 文生图模型自动生成一本「一组一页 · 配图 + 释义 + 词根词缀 + 例句 + 助记 + 画面联想 + 学习笔记」的视觉记忆词汇书，导出 HTML 与 A4 打印 PDF。

| <img src="data/sample_images/O0001_abolish.png" width="260" /> | <img src="data/sample_images/O0062_ambulance.png" width="260" /> |
| :-: | :-: |
| `O0001 abolish` | `O0062 ambulance` |

**Demo（CET-6）**：[下载 PDF →](https://github.com/<you>/vocab-book-gen/releases/latest)（约 146 MB，2035 组，大学英语六级词汇）

---

## 完整流程

```
你的单词列表（一行一个词，或 .xls 词表）
        ↓  scripts/deepseek_vocab_cluster.py
    按词根/语义自动聚组（DeepSeek）
        ↓  scripts/text/gen_text.py
    每组生成释义/例句/词根/助记/考点（DeepSeek）
        ↓  scripts/images/
    每组生成夸张漫画式记忆图（DashScope API 或本地 GPU）
        ↓  scripts/book/
    渲染网页版 HTML + A4 打印版 HTML/PDF
```

从一份单词列表到一本完整词汇书，全程自动，只需要 DeepSeek API key（生图可选阿里云 API，无需 GPU）。

**任何考试/词表都能用**：CET-4、CET-6、考研、IELTS、GRE、SAT、行业词汇……用 `--exam-name` 告诉 AI 语境即可。

---

## 仓库结构

```
vocab-book-gen/
├── data/
│   ├── clusters/
│   │   └── optimized_vocab_clusters.csv   # Demo 词组（CET-6，约 2000 组）
│   ├── sample_images/                     # 两张样图，README 用
│   ├── sample_vocab_text.jsonl            # 前 10 组生成结果，参考字段结构
│   ├── images/                            # 你跑出来的 PNG 落这里（git 忽略）
│   └── vocab_book_text.jsonl              # 文本生成结果（git 忽略）
├── scripts/
│   ├── text/
│   │   └── gen_text.py                    # 1. DeepSeek 生成全部正文文本
│   ├── images/
│   │   ├── prepare_qwen_image_jobs.py     # 2. 生成图片任务 jsonl
│   │   ├── generate_visual_scenes_deepseek.py  # 2a. DeepSeek 改写为无字面拷贝的画面脚本
│   │   ├── generate_images_dashscope.py   # 3. API 生图：DashScope（通义万相）
│   │   ├── run_qwen_stream_worker.py      # 3. 本地生图：Qwen-Image Diffusers worker
│   │   ├── stream_qwen_image_jobs.py      # 3. 本地生图配套投递脚本
│   │   └── run_qwen_batch_images.py       # （旧批量脚本，保留参考）
│   ├── book/
│   │   ├── build_full_book.py             # 4. 网页版 HTML
│   │   ├── build_full_portrait.py         # 5. A4 打印版 HTML（封面 + 中英对照表）
│   │   └── export_pdf.sh                  # 6. Edge headless -> PDF
│   └── （preprocessing utilities，用于从零构建词组 CSV）
│       ├── group_words.py / qwen_group_words.py / deepseek_vocab_cluster.py
│       ├── filter_basic_words.py / filter_by_3500.py
│       └── optimize_vocab_clusters.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 快速开始

```bash
git clone https://github.com/<you>/vocab-book-gen.git
cd vocab-book-gen
python -m venv .venv && source .venv/Scripts/activate    # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env
# 填好 DEEPSEEK_API_KEY；用 API 生图再填 DASHSCOPE_API_KEY
```

---

## 从你的单词列表跑出一本书

你只需要准备一份单词列表，格式随意：一行一个词的 `.txt`，或者任意来源的词表文件。

### Step 0 · 把单词聚组（DeepSeek 自动完成）

把你的单词列表整理成一列，每行一个词，保存为 `data/my_words.txt`（或 `.xls/.csv` 均可）。

```bash
python scripts/deepseek_vocab_cluster.py \
    --input data/my_words.txt \
    --output data/clusters/my_clusters.csv \
    --exam-name "考研"
```

DeepSeek 会按词根/词义/主题把单词自动归组，输出一份聚类 CSV。Demo 里的 CET-6 聚类就是这么跑出来的。

> 如果你的词表已经手工分好组了，可以直接跳到 Step 1，把 CSV 整理成对应格式即可（必填列：`optimized_group_id`、`headword`、`all_memory_words`）。

### Step 1 · 生成正文文本（DeepSeek）

```bash
python scripts/text/gen_text.py \
    --input data/clusters/my_clusters.csv \
    --output data/vocab_book_text.jsonl \
    --exam-name "考研"
```

`--exam-name` 嵌入 AI prompt（CET-4、CET-6、IELTS、GRE 等均可）。2000 组单进程约 6-10 小时，用 `--only-ids` 分片开 4 个进程并行可缩到 1-1.5 小时。

### Step 2 · 生成图片 prompt

```bash
# 推荐：先让 DeepSeek 把每组改写成无字面拷贝的画面脚本，生图质量更好
python scripts/images/generate_visual_scenes_deepseek.py \
    --input data/vocab_book_text.jsonl

python scripts/images/prepare_qwen_image_jobs.py \
    --visual-scenes data/qwen_visual_scenes.jsonl \
    --output data/qwen_image_jobs.jsonl
```

### Step 3 · 生图（两选一）

#### 方案 A · API 生图：阿里云 DashScope（无需 GPU）

```bash
python scripts/images/generate_images_dashscope.py \
    --jobs data/qwen_image_jobs.jsonl \
    --output-dir data/images
```

断点续传（已有 PNG 自动跳过）。想换其他图像 API？拷一份脚本改端点即可，jobs 格式不变。

#### 方案 B · 本地 GPU 生图（Diffusers + Qwen-Image，需 >=48 GB 显存）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors

python scripts/images/run_qwen_stream_worker.py \
    --jobs data/qwen_image_jobs.jsonl \
    --model /path/to/qwen-image \
    --output-dir data/images
```

消费级显卡可在脚本里换成 SDXL-Turbo / FLUX.1-schnell 等小模型。

### Step 4 · 渲染 HTML + 导出 PDF

```bash
# 网页版
python scripts/book/build_full_book.py \
    --title "我的考研词汇书" --exam-label "考研考点"

# A4 打印版（--img-max / --img-quality 控制图片质量和 PDF 大小）
python scripts/book/build_full_portrait.py \
    --title "我的考研词汇书" --exam-label "考研考点" \
    --img-max 800 --img-quality 50

# 导出 PDF（需要 Microsoft Edge；Linux 改 export_pdf.sh 用 chromium）
bash scripts/book/export_pdf.sh
```

输出在 `build/`：`full_book.html`、`full_book_portrait.html`、`full_book_portrait.pdf`。

---

## Demo 词组数据（CET-6）是怎么来的

`data/clusters/optimized_vocab_clusters.csv` 随仓库附带，仅作 demo，就是用上面这套流水线跑出来的：

1. 从原始 CET-6 词表去重 → 5523 词
2. `filter_by_3500.py` 剔除高中熟词，剩约 2900 词
3. `deepseek_vocab_cluster.py` 用 DeepSeek 自动按词根/语义聚组
4. `optimize_vocab_clusters.py` 合并冗余组、补充派生形态

你自己的词表照此流程跑即可，不需要手工整理分组。

---

## 字段结构（jsonl 一行）

```json
{
  "group_id": "O0001",
  "headword": "abolish",
  "page_title": "abolish / abolition, abolitionist",
  "core_meaning": "废除；废止",
  "frequent_translations": ["废除", "废止", "取消"],
  "word_items": [
    {
      "word": "abolish",
      "phonetic": "/əˈbɒlɪʃ/",
      "pos": "vt.",
      "translation": "废除；废止；取消",
      "usage_note": "常搭配制度/法律/习俗，与 cancel/repeal 辨析。",
      "example_en": "...",
      "example_zh": "...",
      "example_en_2": "...",
      "example_zh_2": "..."
    }
  ],
  "root_affix": "abolish 源自拉丁语 abolere……",
  "memory_link": "想象议会大厅一群 abolitionist 高举印章……",
  "mnemonic": "abolish 废除勿迟疑……",
  "study_note": "重音在第二音节……",
  "collocations": ["abolish slavery - 废除奴隶制", "..."],
  "related_recommendations": ["cancel - 取消（具体安排）", "..."],
  "exam_note": "六级阅读/完形高频……"
}
```

---

## 兼容性 / 已知坑

- **PowerShell 显示乱码**：文件永远是 UTF-8 无 BOM，用 Git Bash 或支持 UTF-8 的编辑器打开。
- **PDF 体积**：默认 800px/q50，2000 组约 100-120 MB。降 `--img-quality` / `--img-max` 可进一步缩小。
- **本地生图显存**：Qwen-Image 约需 50 GB 显存；消费级显卡换小模型（SDXL-Turbo、FLUX.1-schnell）。
- **导 PDF 需要 Edge**：Windows/macOS 默认有效。Linux 改 `export_pdf.sh` 里的路径指向 `chromium-browser` 即可。

---


## 协议

代码采用 MIT。生成的词条文本与图片均由 LLM/T2I 模型产生，使用前请自行甄别准确性。
