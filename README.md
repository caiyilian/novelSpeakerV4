# NovelSpeakerV4

**小说对话说话人标注系统** —— 把整本小说中每一处 `「」` 引号的内容，标注出说话人。

```
输入  (novel.txt)                      输出  (labeled.txt)
「这是最后一件了吧？」                   村民
「嗯，这里确实有……七十件。多谢惠顾。」     罗伦斯
```

**评估集**：5 卷 / 7009 条引语 · **目标**：每卷准确率 > 95%

---

## 当前状态（2026-09-24）

| 指标 | 值 |
| --- | --- |
| **最优方案** | 综合两层仲裁（`final_arbiter.py`，分支 `feat/scheme-7-combined`） |
| **五卷准确率** | **94.65%**（方案7）/ 94.64%（方案3）/ 94.45%（方案2，机制最简洁） |
| **距 95% 门槛** | 25–39 条 |
| **单卷用时** | 3.2–3.8 h（五卷并行）→ 端到端约 4 h |
| **模型** | `deepseek-v4-flash`（唯一允许使用的模型） |

> ⚠️ **重要更正**：早期版本曾报告"93.65% 是模型能力上限"——**该结论已被推翻**。原因是评分器存在 bug（只对输出侧做"无人称引语→非人物发声"映射，答案侧未映射），导致 56 条被误判为错。修正后方案7 达 94.65%，且剩余错误中 **227 条属于"共识盲区"**（两个通道给出相同标签但都错，分歧检测器根本不触发）——这是**机制覆盖不足**，不是能力上限。

---

## 快速开始

```bash
# 依赖（可选，仅桌面控制台需要）
python -m pip install -r requirements-gui.txt

cd annotator

# 1) 单卷标注（连续多轮基线）—— 在 master 上可直接运行
python cont_agent.py --volume 1 --tag contfull

# 2) 整卷通道（第二信号源）—— 在 master 上可直接运行
python evidence_annotate.py --volume 1 --with-ledger --tag evledger

# 3) 评分（必须用修正后的评分器，双侧归一）
python score_fixed.py --evidence frozen

# 4) 分歧仲裁（当前最优方案之一，需先切分支）
git checkout feat/scheme-2-arbiter
python arbiter_agent.py --volume 1 --tag arb
```

**模型池配置**：`annotator/probe_dsv4flash.py` 的 `load_pool()` 读取 API 池（20 个账号）。
**角色账本**：`annotator/twopass_annotate.py` 生成，是各裁决器的共同输入。
**运行环境**：必须用项目 venv（`binaries/python/envs/default/`），managed Python 3.13 缺 `requests`。

---

## 方案与结论

围绕"提升准确率"共实现并五卷全量测试了 **8 个方案**：

> ⚠️ **脚本位置**：下表 8 个方案中，**只有基线 `cont_agent.py`、整卷通道 `evidence_annotate.py`、评分器 `score_fixed.py` 在 `master` 上**；
> 其余 5 个方案脚本位于各自的功能分支（**未合入 master**）。查看请先 `git checkout <分支>`：
>
> | 方案 | 脚本 | 分支 |
> | --- | --- | --- |
> | 综合两层仲裁 | `final_arbiter.py` | `feat/scheme-7-combined` |
> | 三通道仲裁 | `arbiter3_agent.py` | `feat/scheme-3-three-pass-arbiter` |
> | 分歧仲裁 | `arbiter_agent.py` | `feat/scheme-2-arbiter` |
> | 精确契约 | `arbiter5_agent.py` | `feat/scheme-5-precise-contract` |
> | 对抗裁决 | `refuter_agent.py` | `feat/scheme-4-refuter` |
> | 场景块 | `scene_block_agent.py` | `feat/scheme-1-scene-block` |
> | 判据修正 | `np_refine_agent.py` | `feat/scheme-6-attributable-np` |
>
> 所有分支均已推送到 `github.com/caiyilian/novelSpeakerV4`。

| 排名 | 方案 | 脚本 | 五卷 | 结论 |
| ---: | --- | --- | ---: | --- |
| 1 | **综合两层仲裁** | `final_arbiter.py` | **94.65%** | 当前最优 |
| 2 | 三通道仲裁 | `arbiter3_agent.py` | 94.64% | 与最优差 1 条 |
| 3 | 分歧仲裁 | `arbiter_agent.py` | 94.45% | 机制最简洁 |
| 4 | 精确契约 | `arbiter5_agent.py` | 93.69% | 健壮性改进 |
| 5 | 对抗裁决 | `refuter_agent.py` | 93.47% | 与分歧仲裁互补 |
| 6 | 场景块 | `scene_block_agent.py` | 93.04% | 场景级视野 |
| 7 | 判据修正 | `np_refine_agent.py` | 92.97% | 事后补救 |
| 8 | 整卷通道 | `evidence_annotate.py` | 92.82% | 第二信号源 |
| — | cont 基线 | `cont_agent.py` | 92.18% | 逐条滚动 |

**三条被数据验证的核心结论**：

1. **分歧仲裁是唯一稳定有效的机制** —— 两个独立通道（逐条滚动 vs 整卷+账本）的标签分歧 = 真实信息增量，远优于同源投票。
2. **新增信号源必须同时满足「不同源」和「足够准」** —— 低质量第三通道引入的分歧中噪声是信号的 5.5 倍（实测）。
3. **改动评测口径时，评分器必须同步审计** —— 本项目曾因此产生 56 条的评分偏差。

**完整记录**：`docs/最终汇总_八方案全记录_2026-09-23.md`（含逐卷用时、产物位置索引）

---

## 当前瓶颈与下一步

剩余错误混合**四种成因**，其中最大一块是机制盲区：

| 成因 | 量级 | 说明 |
| --- | ---: | --- |
| **共识盲区** | **227 条** | 两通道标签相同但都错，**分歧检测器不触发** |
| 双方都错 | 285 条 | 含上项交集，不可相加 |
| 局部标签写法 | ~60 条 | 同一实体在不同段落称谓不同 |
| 群体粒度 | ~57 条 | 群体被写成过宽泛称 |

**下一步方向**（尚未实施）：

1. **答案口径统一** —— 第一卷答案 **51% 是多标签**（505 条 `贤狼赫萝|赫萝`）。规则：**有名字用名字，没名字用职业**。其中 665/690 条可由账本规则自动消解。
2. **场景实体层** —— 加入"共识风险触发器"，覆盖 227 条盲区（而非再叠一层泛化裁决）。
3. **评分口径收紧** —— 多标签答案改为唯一标签后，需重算所有历史结果。

**第一卷答案审阅流水线**（已就绪）：

| 文件 | 用途 |
| --- | --- |
| `docs/审阅_vol1_文件A_待核清单.md` | 错误清单（46 条模型≠答案 + 25 条多标签） |
| `docs/审阅_vol1_文件B_小灵判断.md` | AI 助手逐条判断（附原文证据） |
| `docs/审阅_vol1_大模型审核结果_2026-09-24.txt` | 独立大模型判断 |
| `docs/审阅_vol1_差异对比_小灵vs大模型_2026-09-24.md` | 交叉核实（双方各纠正对方 3–7 条） |
| `docs/审阅_vol1_最终人工审阅版.html` | **交互式审阅页**（64 条判断题，17 条分歧高亮） |

---

## 项目结构

```
novelSpeakerV4/
├── annotator/              # 标注与实验脚本（核心）
│   ├── cont_agent.py              # 基线：连续多轮逐条标注（master）
│   ├── evidence_annotate.py       # 整卷通道 / 第二信号源（master）
│   ├── twopass_annotate.py        # 角色账本生成（master）
│   ├── score_fixed.py             # 评分器：双侧归一（master）
│   ├── make_error_list.py         # 审阅流程：生成文件A（master）
│   ├── make_final_review.py       # 审阅流程：生成交互网页（master）
│   ├── revise_answers.py          # 答案口径修订（master）
│   └── results/                   # 标注产物（不入 git，按日期归档）
│   # 其余方案脚本（arbiter_agent / final_arbiter / refuter_agent /
│   #   arbiter3_agent / arbiter5_agent / scene_block_agent /
│   #   np_refine_agent）在各自 feat/scheme-* 分支，未合入 master
│
├── data/                   # 语料与答案
│   ├── novel.txt                  # 第 1 卷原文
│   ├── answers.txt                # 第 1 卷参考答案
│   ├── evidence_vault.json        # 已验证身份库（评分器依赖）
│   └── volume{2..5}/              # 第 2-5 卷 novel.txt + answers.txt
│       # 注意：第 1 卷直接用 data/ 根目录（代码里 VOLUME_DIRS[1] = data/）
│
├── src/                    # 旧系统（24 个模块，约 16900 行；保留供参考）
│   └── run_label.py               # 评分核心（_validation_lenient_match 等）
│
├── backup/                 # 历史产物与冻结证据库（不入 git）
├── docs/                   # 报告与审阅文档（44 份）
├── tmp/                    # 临时实验（不入 git）
│
├── launch_control_center.cmd      # 桌面控制台启动
└── build_control_center_exe.cmd   # 打包 EXE
```

---

## 标注规则（判定口径）

**说话人归属**：

1. 角色当场说出的话 → 标该角色
2. 内心独白（原文有「心想」「暗自说」「脑海里」等心理动词）→ 归该角色
3. 群体发言 → 标群体名
4. 以下标 `无人称引语`：
   - 纯环境声、动物声、拟声词
   - **被当作语言材料谈论的词句**（原文有「这句话」「这两个字」「这个称呼」「单字」等**元语言标记**）
   - 叙述者给神态/姿态配的字幕式引号
5. 短语气词、应答词、省略号（「唔。」「咦？」「……」）**仍是发言**

**标签写法**：**有名字的用名字，没有名字的用职业或身份**；同一角色全卷统一标签。

> 该口径由 `docs/重构准备三_口径裁定与三阶段流水线_2026-09-20.md` 的多模型裁定确定。

---

## 桌面控制台（可选）

```cmd
python -m pip install -r requirements-gui.txt
launch_control_center.cmd
```

首次启动需选择 API key 文件（仅保存路径）。控制台以隐藏子进程启动各卷、关闭窗口后驻留系统托盘；支持继续/重试（保留断点）与重启（去重备份后 `--reset-state`）。

打包单文件 EXE：

```cmd
build_control_center_exe.cmd    # 输出 dist\NovelSpeakerControlCenter.exe
```

---

## 环境说明

- Python 3.13（managed）或 3.8；实验脚本需 `requests`
- **必须使用项目 venv**：`C:/Users/Administrator/.workbuddy/binaries/python/envs/default/`
  （managed Python 3.13 缺 `requests`）
- 产物（`annotator/results/`、`tmp/`、`data/`、`backup/`）**均不入 git**，本地按日期归档

## License

MIT
