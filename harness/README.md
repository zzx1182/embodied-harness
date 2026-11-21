# harness

配合「模型建议、执行校验与可靠回执」「π₀.₅、Direct 与 Hybrid 控制」「具身 Agent 评测、扰动与任务编排」三课使用。

让大模型通过 move_eef 和 give_up 两个工具控制 LIBERO 机械臂。模型只提建议，执行器校验、执行、记账，任务成不成功由环境判。「π₀.₅、Direct 与 Hybrid 控制」在同一个包上接 π₀.₅ 策略服务，加 accept_vla 工具，用 --mode 切换 Direct、VLA-only、Hybrid 三种模式。「具身 Agent 评测、扰动与任务编排」再加两种模式，编排（模型只下子指令，动作全由 π₀.₅ 出）和写死顺序，以及把多批结果汇成评测报告的 eval_report.py。

## 目录

| 文件 | 作用 |
|---|---|
| harness/contracts.py | 四个契约和执行边界 Spec |
| harness/validator.py | 四道校验，返回 None 或拒绝记录 |
| harness/ledger.py | SQLite 账本 |
| harness/executor.py | 单写执行器：去重、观测新鲜度、候选登记与核对、先落盘再执行、UNKNOWN |
| harness/runtime.py | 打包上下文、调模型、预算、终止 |
| harness/models.py | OpenAI 兼容接口、Codex CLI、脚本模型 |
| harness/envs.py | LIBERO 适配器和假环境 |
| harness/report.py | 从账本重算报告，含部分完成和误验收审计 |
| harness/aggregate.py | 多批结果汇总、失败主因分类、按 case 配对（「具身 Agent 评测、扰动与任务编排」） |
| harness/policy.py | openpi WebSocket 策略客户端（「π₀.₅、Direct 与 Hybrid 控制」） |
| harness/runner.py | 跑一条 episode 的公共入口 |
| policy_manifest.py | 写策略 manifest：checkpoint 逐文件校验结果、norm_stats 哈希、预处理版本、openpi 与 lerobot commit、控制器规格、服务元数据 |
| probe_frozen.py | 冻结快照探针：同一份观测连发两次正常请求、一次故意做错的预处理，记形状、量程和候选偏移 |
| samples/eval_runs/ | 参考环境一轮完整评测的 165 条原始记录，没有显卡也能跑 eval_report.py（见 samples/README.md） |
| run_episode.py | 跑一条 episode，--mode direct / vla / hybrid / orchestrate / scripted |
| run_matrix.py | 按任务、seed、初态批量跑，写评测声明和逐 case 表 |
| eval_report.py | 把若干批 run_matrix 的输出汇成一份报告（「具身 Agent 评测、扰动与任务编排」） |
| tests/test_regression.py | 十条回归用例 |
| tests/test_extensions.py | 五条扩展用例：中途取消、接口失败不算拒绝、编排与宣布审计、写死顺序、失败分类 |
| tests/test_infra.py | 七条接口与预算用例：配置错误立即停机、限流才退避、逐段进度、VLA-only 只受控制步约束、策略调用上限、401/429 区分、Codex 临时目录隔离 |
| tests/test_candidates.py | 六条候选核对用例：新鲜候选放行；旧帧候选、前缀越界（含剩余预算）、纠偏后接旧尾部、策略换了、没登记过的候选一律拒绝 |
| tests/test_batches.py | 六条批次与汇总用例：换了模型不许续跑、重复的 case 报错、同一目录给两遍只算一次、一批里混了两种模型报错、配对表只留模式齐全的 case |
| requirements-test.txt | 不需要 GPU 的那部分依赖（numpy、scipy、pillow），单独下载这个附件、还没装 LIBERO 环境时用 |

## 先跑回归用例（不需要 GPU，不需要模型）

已经装好 LIBERO 环境的，在那个环境里直接跑。单独下载这个附件、还没有环境的，先建一个 Python 3.12 的虚拟环境装三个包：

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
```

然后跑用例，再用样例数据出一份报告：

```bash
python -m unittest -v tests/test_regression.py tests/test_extensions.py tests/test_candidates.py tests/test_infra.py tests/test_batches.py
python eval_report.py samples/eval_runs --out outputs/eval_report_sample
```

用例全部显示 ok，最后一行是 `Ran 34 tests` 和 `OK`；报告那条命令打印「165 条 episode，报告写在 outputs/eval_report_sample/report.md」。

## 再跑一条真的 episode

```bash
export MUJOCO_GL=egl
export HARNESS_MODEL_BASE_URL=https://你的接口/v1
export HARNESS_MODEL_NAME=模型名
export HARNESS_MODEL_API_KEY=密钥
python run_episode.py --model openai --suite libero_object --task 0 --max-calls 30 --out outputs/direct_object0
```

输出目录：

| 文件 | 内容 |
|---|---|
| ledger.sqlite | requests、request_events、session_meta、model_calls、policy_calls、scoring、declarations 七张表，scoring 和 declarations 存目标谓词真值，只给评测用，不进模型上下文 |
| segments/000/ | 每个决策段：两张观测图、context.json、model_raw.json、decision.json、validation.json、request.json、receipt.json |
| report.json、report.md | 从账本重算的 episode 报告 |

重算报告：

```bash
python -c "from harness.report import write_report; write_report('outputs/direct_object0/ledger.sqlite','outputs/direct_object0')"
```

## 改一个校验条件（第八节）

限幅在 harness/contracts.py 的 Spec 里，payload_hash 对比和 expected_obs_seq 对比在 harness/executor.py 和 harness/validator.py 里。改一处，跑回归，看哪条变红，改回来。

## 三种模式与批量运行（「π₀.₅、Direct 与 Hybrid 控制」）

先按「π₀.₅、Direct 与 Hybrid 控制」第一节起 π₀.₅ 策略服务，再把 openpi 客户端加进路径。

```bash
export HARNESS_OPENPI_CLIENT_SRC=/path/to/openpi/packages/openpi-client/src
python run_episode.py --mode vla --policy-port 43861 --suite libero_object --task 0 --out outputs/vla_object0
python run_episode.py --mode hybrid --model openai --policy-port 43861 --suite libero_object --task 0 --out outputs/hybrid_object0
python run_matrix.py --mode vla --policy-port 43861 --suite libero_object --tasks 0-9 --seeds 0 --out outputs/matrix_vla/libero_object
```

run_matrix.py 已有 report.json 的 case 会跳过，中断后重跑同一条命令即可续上。续跑时模式、任务、seed、预算、模型（`--model` 和 `HARNESS_MODEL_NAME` 等环境变量）和第一次不一样会被拒绝；加上 `--policy-manifest outputs/policy_manifest.json`，换了权重也会被拒绝。eval_report.py 遇到两批里有同一个 case 的同一种模式会报错停下，只把要用的那一批传给它。LIBERO-Pro 的扰动套件（libero_object_swap、libero_object_task 等）在它自己的 venv 里用同样的命令跑，安装步骤见「π₀.₅、Direct 与 Hybrid 控制」第六节。

模型接口失败（容量不足、断网、额度用完）不算模型的拒绝，也不占调用预算。runtime 等 60 秒在同一个观测上重问，连续 30 次失败（约半小时）以 infra:model_unavailable 停机，次数记在报告的 model_errors 里。失败调用的耗时和退避等待记在 infra_wait_s，不计入墙钟预算。

## 编排、写死顺序与评测报告（「具身 Agent 评测、扰动与任务编排」）

```bash
COMMON="--suite libero_10 --tasks 0 --seeds 0 --init 0-2 --max-control-steps 400 --max-calls 30 --max-wall-s 3000 --policy-port 43861"
PLAN="pick up the alphabet soup and place it in the basket::20||pick up the tomato sauce and place it in the basket::20"
python run_matrix.py --mode scripted --plan "$PLAN" $COMMON --out outputs/orchestration/scripted
python run_matrix.py --mode orchestrate --model openai $COMMON --out outputs/orchestration/orchestrate
python eval_report.py outputs/matrix_vla outputs/orchestration --out outputs/eval_report
```

编排模式的工具是 run_skill（子指令加 1 到 5 个动作块）、declare_subgoal_done、finish、give_up。宣布时的目标谓词真值写进 declarations 表，report.json 里的 false_acceptances 就是宣布了但真值为假的次数。eval_report.py 按套件、任务、seed、初态、控制步上限五样相同配对，失败主因的判定顺序写在 harness/aggregate.py 开头。
