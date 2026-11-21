# first_loop · 「LIBERO 仿真环境与第一条可执行动作」附件

在 LIBERO 里跑第一条不带模型的闭环：取观测、发固定动作、读末端位姿、拦坏动作、写账本。

## 前提

按「LIBERO 仿真环境与第一条可执行动作」第一节装好 venv 和 LIBERO，在激活的环境里、没有显示器的机器上先 `export MUJOCO_GL=egl`。
把这个目录解压到你的项目文件夹里（教程里叫 embodied-harness），在这个目录下运行。

## 按顺序跑

```bash
python 01_reset_observe.py                 # 第二节：reset，过滤白名单，存两张图
python 02_fixed_action.py --dx 0.2 --steps 15   # 第三节：固定动作连发 15 步
python 02_fixed_action.py --dx 0.2 --steps 5    # 第四节：同一动作只发 5 步
python 03_bad_actions.py --demo-clipping   # 第五节：超量程和 NaN 被拦下
python 04_show_ledger.py                   # 第六节：把账本打印成对照表
```

产物都在 `outputs/first_loop/`：

| 文件 | 内容 |
|---|---|
| agentview.png、wrist.png | 两路 256×256 画面 |
| observation.json | 七项白名单观测，图像只记形状 |
| dropped_keys.json | 被过滤掉的键名 |
| ledger.jsonl | 账本，一条请求一行，只追加 |

## 改什么

- 换任务：`--suite libero_spatial --task 0`，三个套件各有 10 个任务。
- 换动作：`--dx 0.1`，范围 [-1, 1]，只动 x 维。想动别的维度改 02 里的 action 列表。
- 想重新开始：删掉 `outputs/first_loop/ledger.jsonl`，request_id 会从 0001 重新计数。

## 文件

| 文件 | 作用 |
|---|---|
| common.py | 建环境、白名单过滤、check_action、读末端、账本 |
| 01_reset_observe.py | 第二节 |
| 02_fixed_action.py | 第三、四节 |
| 03_bad_actions.py | 第五节 |
| 04_show_ledger.py | 第六节 |
