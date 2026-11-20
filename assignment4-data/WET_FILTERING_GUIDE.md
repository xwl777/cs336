# WET 并行过滤：代码讲解与实际结果

## 1. 这次实际完成了什么

把一份真实的 Common Crawl WET 文件读成文档，用两个进程运行已有分类器，输出筛选和脱敏后的训练文本。逐步统计、每篇文档的判定记录、随机样本与人工判读都保留在本地，方便复查。

本次输入是 `data/example.warc.wet.gz`，不是题目共享目录下的 2,500 份数据。本机没有该共享数据目录，所以先用已经下载的现有数据实跑。它还没有做英文筛选，因此新增英文概率至少 0.7 的步骤，与题目起点对齐。

结果：19,637 篇输入，保留 1,659 篇，保留率 8.45%；在通过英文筛选的 5,842 篇中，保留率约 28.40%。总耗时 15.63 秒，512 篇发生脱敏，最终保留约 1,309 万字符。字符数不等于 token 数。

这一步产出训练文本，不包括 GPT-2 分词、模型训练或 Paloma 困惑度评测。没有复制或使用 Paloma 验证文本，也不能据此声称验证困惑度已经改善。

## 2. 代码在哪里，怎样运行

- [逐篇过滤逻辑](cs336_data/filtering.py)：只管一篇文档依次通过哪些检查。
- [并行脚本](scripts/filter_wet.py)：读取 WET、分批、启动进程、汇总和输出。
- [流水线测试](tests/test_filter_pipeline.py)：检查过滤顺序、脱敏、跨分片去重与统计。

在 `/Users/lwx/cs336/assignment4-data` 下运行：

```bash
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run --offline python -m scripts.filter_wet \
  data/example.warc.wet.gz \
  --output data/filtered_wet_baseline \
  --workers 2 \
  --chunk-size 1000
```

上面是本次实际命令，输出目录已经存在。再次运行时换一个新的输出目录，避免把不同实验的文件混在一起。

也支持多个文件或目录：

```bash
uv run python -m scripts.filter_wet /shared-data/english-wet-data \
  --output data/filtered_english_run --workers 2 --chunk-size 1000
```

目录输入会查找该目录直接包含的 `*.warc.wet.gz` 文件。`--workers` 是进程数，不是文件数；每个进程缓存各自的模型，两个有害内容模型每个约 1 GB，所以不能只按 CPU 核心数盲目增加进程。

## 3. 一篇文档怎样过关

看 `filtering.py` 的 `filter_document(row, settings)`。`row` 包含网页 URL、来源文件、文档编号和正文 `text`；函数返回处理后的文档及审计记录。如果被删除，第一个返回值是 `None`。

依次执行以下步骤：

1. **nonempty**：正文不能全是空白。
2. **english**：fastText 的首选语言必须是 `en`，该标签置信度至少 0.7。
3. **length**：按空白分词，词数在 50 到 100,000 之间。
4. **alphabetic**：至少 80% 的词包含一个字母字符，减少数字或符号堆积。这是一个基础规则，不是完整 Gopher 规则集。
5. **nsfw**：只有模型返回 `nsfw` 且该标签分数至少 0.9 才删除。
6. **toxic**：只有模型返回 `toxic` 且该标签分数至少 0.9 才删除。
7. **quality**：`quality_score(text)` 至少 0.5。这里始终取高质量类别的分数。
8. **pii**：依次遮盖邮箱、美国常见电话号码和 IPv4，记录替换次数。这一步修改文本，不删除文档。
9. **exact_document_dedup**：所有进程结束后，跨分片比较脱敏后全文的哈希。比较前仅合并空白；保留第一份，删除后续重复文档。

**不要混淆标签置信度与有害概率。** `non-toxic, 0.99` 是模型确信无毒，不能因为分数大就删除。质量函数则专门返回高质量类分数，数值越高越倾向保留。

每一步失败后立即返回，避免给明显不符合要求的文档继续运行昂贵分类器。因此删除原因是“第一个失败的步骤”，并不代表它只存在这一种问题。改变顺序会改变各步骤的删除归因。

内部的 `record(stage, passed, start)` 做三件事：保存步骤名、记录通过与否、计算耗时；失败时还会设置 `removed_by`。子进程稍后把这些记录累加为统计。

## 4. 为什么只有一个文件也能并行

如果只把“一个 WET 文件”作为“一个任务”，本次只能用到一个进程。`prepare_chunks()` 先逐条读取 WET，每 1,000 篇写成一个临时 JSONL 分片；本次得到 20 个分片。

FastWARC 只遍历 `conversion` 记录，`record.reader.read()` 读取正文；WARC 的元数据头不被当作训练文本。读取时使用 UTF-8，损坏字节替换为占位字符。

然后 `ProcessPoolExecutor` 把分片交给两个子进程。`process_chunk()` 不把全部正文通过进程通信来回传输，而是将结果写到分片文件，只把计数和少量样本返回父进程。分类器第一次使用时加载，之后同一进程复用缓存。

这里使用 `spawn`，子进程独立启动，不继承其他库的线程状态。每个分片有单独的输出文件，没有多个进程同时写同一个文件。

最后 `merge_outputs()` 按原分片顺序合并。这里必须做一次全局去重：同一篇文章可能同时出现在第 1 和第 18 个分片，只在各子进程内部去重会漏掉。

本次采用全文精确去重，没有默认启用逐行删除和 MinHash 近似去重。前者可能删掉有用的共用短句，后者需要额外开销与阈值校验；这份结果不能称为“已完成所有近似去重”。

## 5. 输出文件如何使用

输出目录：`data/filtered_wet_baseline/`。

- `filtered.jsonl.gz`：最终保留的文档，每行一条 JSON，包含 `text`、`url` 和来源等字段。建议后续分词使用它，文档边界明确。
- `training.txt.gz`：方便直接阅读的纯文本，各篇之间用两个换行分隔；这不是 GPT-2 token 二进制文件。
- `summary.json`：全局统计、阈值、阶段计数及耗时。
- `audit/*.jsonl.gz`：每篇文档经过哪些阶段、模型结果、删除原因和替换次数。
- `dedup_audit.jsonl.gz`：通过前面步骤的文档，最终是否因全文重复被删除。
- `review_samples.json`：5 篇保留、5 篇删除、5 篇脱敏前的样本，以及每类删除原因的额外样本。包含正文，便于复核。
- `manual_review.json`：对随机抽出的 5 篇保留及 5 篇删除文档的中文判读。
- `serial_parallel_check.json`：串行与并行结果一致性的核对。
- `chunks/` 和 `parts/`：输入与处理结果的中间分片，便于排错。

## 6. 统计中的三种比例

例如长度过滤删除了 467 篇：

- **本步删除率**：467 / 5,842，约 7.99%。分母是进入长度过滤的篇数。
- **占原始输入比例**：467 / 19,637，约 2.38%。
- **占全部删除比例**：467 / 17,978，约 2.60%。题目要求的“被丢弃样本中各步骤的贡献”主要看这一列。

PII 修改但不删除，所以删除率为零；应单独看修改文档数和替换次数。

## 7. 耗时应该怎样理解

本机为 macOS arm64，16 GiB 内存，Python 3.12.13，双进程，每批 1,000 篇。这是一次本地实测，不是 GPU 测试。

- 读取 WET 与准备分片：2.49 秒。
- 启动子进程、加载模型和并行过滤：12.16 秒。
- 合并、全文去重与输出：0.98 秒。
- 总计：15.63 秒，约每秒 1,256 篇，按输入压缩体积约 4.06 MB/s。

`worker_seconds` 是各进程内累计的步骤用时，含懒加载模型的开销；两个进程同时工作，因此把它们相加不是整次运行的墙钟时间。阶段用时也不覆盖全部 JSON、压缩、调度和文件写入开销。

另以相同阈值跑了完整的单进程对照：19.42 秒。双进程本次约快 1.24 倍。两次的阶段计数一致，`filtered.jsonl.gz` 与 `training.txt.gz` 解压后的 SHA256 都一致。只有一次对照，不能当作稳定的平均加速比；操作系统文件缓存和并发负载也会影响结果。

若假设 2,500 份文件都与本次样本大小、语言构成和处理成本相同，线性估算约 10.86 小时。题目提供的数据已筛过英文，会有更高比例进入后续模型，不能把这个估算当作该数据集的实测结果。

本次读取了同批次的官方 [WET 路径清单](https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/wet.paths.gz)，其中有 100,000 个 WET 文件；在相同样本假设下约需 18.10 天。这指 CC-MAIN-2026-12 单次 crawl 的条件估算，不是 Common Crawl 历年全部数据。没有下载整个 crawl，估算不含网络传输，也未计入全局去重集合随语料增大后的内存成本。

## 8. 抽样检查后学到了什么

随机抽样使用种子 42：对每个唯一文档编号计算一个固定的伪随机排序值，保留排序最小的 5 篇。各分片先保留局部前 5，再合并取全局前 5，等价于在对应集合中整体抽样，不按分片大小平均分配名额。保留样本从最终去重后的集合抽取。

5 篇删除样本多数落在英文步骤，是因为这份原始 WET 本来就有约 70% 不满足英文阈值。为避免只看到语言问题，还额外保留各删除步骤的两个样本；这些额外样本不是第二组均匀随机样本。

额外检查发现 LyricsGaps 的英语歌曲学习页被 toxic 分类器以约 0.966 的分数删除，结合教学语境属于值得复核的误判。另一个被 toxic 删除的页面则确实是成人内容，说明简单提高 toxic 阈值也可能放过不希望保留的内容，不能只为修复一个误判就直接调参。

本次只完成 0.5 质量阈值的基线数据版本，以及串行/并行一致性对照；没有据五篇样本重新训练分类器或宣称找到了最优阈值。后续应扩充有代表性的人工标签、清理页面模板、改善国际电话号码识别，再通过固定模型训练的 Paloma 困惑度比较各数据版本。保留数据里仍有噪声，不能把每一篇都视为高质量正文。

## 9. 测试与现有代码状态

本次流水线的 4 项测试全部通过，验证首次失败归因、负类高置信度不会误删、脱敏、质量阈值和跨分片去重。恢复 MinHash 实现后，连同精确行去重和 MinHash 相关测试，共 13 项通过。

初次回归发现 `cs336_data/deduplication.py` 缺少 `minhash_deduplication`，导致 5 项 MinHash 测试报导入错误；随后按用户要求恢复该函数及其辅助函数，这 5 项测试均已通过。并行流水线仍使用自己的全文哈希去重，没有改成调用 MinHash；原过滤脚本、过滤数据与统计文件均校验未变。这里仅报告相关的 13 项回归测试，不代表整个作业测试全部通过。

## 10. Assignment 5 普通目录转换

`/Users/lwx/cs336/assignment5-alignment/.git` 已完整移动到 `/private/tmp/cs336-assignment5-git-backup-20261019/git-metadata`，没有删除源码。转换前后校验了 257 个工作文件，内容一致；工作文件 SHA256 清单也保存在同一备份目录。

现在该目录的 Git 根目录为 `/Users/lwx/cs336`，不再是独立仓库。本次没有提交或暂存该目录。原提交历史仍在备份里，临时目录不适合长期归档；如需长期保留历史，请将备份另存到持久位置。

## 11. 各过滤步骤实测统计

下列步骤按实际执行顺序列出；进程累计耗时与整次墙钟耗时口径不同。

- **非空检查**：进入 19,637，保留 19,637，删除 0；本步删除率 0.000%，占全部删除的 0.000%，累计步骤耗时 0.015 秒。
- **英文筛选**：进入 19,637，保留 5,842，删除 13,795；本步删除率 70.250%，占全部删除的 76.733%，累计步骤耗时 8.317 秒。
- **长度筛选**：进入 5,842，保留 5,375，删除 467；本步删除率 7.994%，占全部删除的 2.598%，累计步骤耗时 0.111 秒。
- **含字母词比例**：进入 5,375，保留 5,108，删除 267；本步删除率 4.967%，占全部删除的 1.485%，累计步骤耗时 0.985 秒。
- **NSFW**：进入 5,108，保留 5,105，删除 3；本步删除率 0.059%，占全部删除的 0.017%，累计步骤耗时 5.055 秒。
- **有毒言论**：进入 5,105，保留 5,103，删除 2；本步删除率 0.039%，占全部删除的 0.011%，累计步骤耗时 5.816 秒。
- **质量分类**：进入 5,103，保留 1,661，删除 3,442；本步删除率 67.451%，占全部删除的 19.146%，累计步骤耗时 1.416 秒。
- **PII 脱敏**：进入 1,661，保留 1,661，删除 0；本步删除率 0.000%，占全部删除的 0.000%，累计步骤耗时 0.675 秒。
- **全局全文去重**：进入 1,661，保留 1,659，删除 2；本步删除率 0.120%，占全部删除的 0.011%，累计步骤耗时 0.088 秒。

脱敏阶段共替换邮箱 585 处、电话号码 590 处、IPv4 24 处；去重后仍有 512 篇被修改的文档。

## 12. 随机抽出的 5 篇保留与 5 篇删除文档


### 保留文档


**1. 菲律宾高校合作新闻**（文档编号 19376）

来源：[https://www2.spamast.edu.ph/forging-academic-partnerships-spamast-and-siliman-university-sign-mou-and-launch-lecture-series/](https://www2.spamast.edu.ph/forging-academic-partnerships-spamast-and-siliman-university-sign-mou-and-launch-lecture-series/)

> This partnership aims to foster knowledge exchange, collaborative research, and innovative educational programs

有完整的合作签约与讲座报道，适合提供教育和环境领域的英文叙述；但无障碍声明和导航占比较高，应继续清理模板。

高质量类别分数：0.518。


**2. 贝斯和弦说明**（文档编号 5405）

来源：[https://fretastic.com/bass/chord/e-sharp-sus4-chord-caged](https://fretastic.com/bass/chord/e-sharp-sus4-chord-caged)

> A E# Suspended Fourth chord ... contains the notes E#, A# and B#.

包含音符与音程知识，可作为专业领域短文本；只有81词且菜单比例较高，不应仅因质量分数过0.5就视为优质长文。

高质量类别分数：0.527。


**3. 英语播客简介**（文档编号 18088）

来源：[https://www.spiked-online.com/podcast-episode/trans-ideology-is-an-assault-on-reality/](https://www.spiked-online.com/podcast-episode/trans-ideology-is-an-assault-on-reality/)

> Bev Jackson talks to Brendan O’Neill about the LGB Alliance and the fight for gay rights.

主要是节目简介、导航及订阅推广，实质正文有限，语言建模价值偏低；涉及社会议题的立场表达并不自动等于有毒言论，现有低毒分也不保证内容中立。

高质量类别分数：0.508。


**4. 银行与伊斯兰金融新闻聚合**（文档编号 15371）

来源：[https://www.islamicfinance.de/?q=taxonomy/term/931](https://www.islamicfinance.de/?q=taxonomy/term/931)

> Talks are underway in Abu Dhabi for a possible tie-up between Abu Dhabi Islamic Bank with First Abu Dhabi Bank

多段新闻摘要有较完整的金融叙述，保留基本合理；重复新闻标题、分类标签与侧栏仍然较多，整篇去重无法去掉这类局部冗余。

高质量类别分数：0.756。


**5. 滑雪教练个人经历博客**（文档编号 17578）

来源：[https://www.roddywillis.com/2018/08/roddy-willis-30-ski-seasons-loving-skiing-mountains/](https://www.roddywillis.com/2018/08/roddy-willis-30-ski-seasons-loving-skiing-mountains/)

> The common thread across the last 30 years is my love of the skiing and the mountains

正文连贯，描述工作经历与滑雪教学，适合英文语言建模；仍混有预订入口与重复图片说明，且瑞士格式的电话号码未被美国号码规则覆盖。

高质量类别分数：0.566。


### 删除文档


**1. 俄语亚美尼亚旅游行程**（文档编号 10573）

来源：[https://sputniktravel.am/tour/view/55/rus/](https://sputniktravel.am/tour/view/55/rus/)

> ПАСХА В АРМЕНИИ

英文过滤删除，主要内容确实是俄语，对本题英文训练目标而言合理；不代表旅游文章本身质量差。

触发步骤：`english`。


**2. 日语汽车参数目录**（文档编号 13233）

来源：[https://www.carsensor-edge.net/catalog/mercedes-benz/c_class_wgn/F002M008/](https://www.carsensor-edge.net/catalog/mercedes-benz/c_class_wgn/F002M008/)

> メルセデス・ベンツ Cクラスワゴン

英文过滤删除，主体为日语，符合英文训练目标；页面本身仍可供日语汽车领域模型使用。

触发步骤：`english`。


**3. 美国路易斯安那州工资表**（文档编号 2346）

来源：[https://apps01.civilservice.louisiana.gov/onestopjobinfo/JobViewPayLevels.aspx?Schedule=SS&PayLevel=417](https://apps01.civilservice.louisiana.gov/onestopjobinfo/JobViewPayLevels.aspx?Schedule=SS&PayLevel=417)

> State Civil Service Pay Grid / Social Services Schedule

英文置信度只有0.473，未达0.7而删除；人工看是英文夹大量数字，属于语言筛选的误拒，表格对当前连续英文文本目标的收益另需评估。

触发步骤：`english`。


**4. 英国办公椅商品页**（文档编号 18754）

来源：[https://www.ukofficedirect.co.uk/product/houston-heavy-duty-task-operator-chair-black-mesh-back-black-fabric-seat/kdm176](https://www.ukofficedirect.co.uk/product/houston-heavy-duty-task-operator-chair-black-mesh-back-black-fabric-seat/kdm176)

> Buy Houston Heavy Duty Task Operator Chair Black Mesh Back Black Fabric Seat

质量分数0.254而删除，页面包含大量重复商城导航、规格与购物操作，作为偏连贯文章的数据集可合理降低其权重；商品说明也有知识价值，不能断言对C4验证必然无用。

触发步骤：`quality`。

高质量类别分数：0.254。


**5. 德语论坛用户资料页**（文档编号 18372）

来源：[https://www.telefon-treff.de/user/4114-smarti/](https://www.telefon-treff.de/user/4114-smarti/)

> Smarti - Mitglieder - Telefon-Treff

英文过滤删除，主体是德语且以登录、导航和空留言板为主，在英文训练目标下删除合理。

触发步骤：`english`。
