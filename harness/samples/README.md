# 样例评测数据

参考环境上一轮完整评测的原始记录，一共 165 条 episode，没有显卡也能拿它跑评测报告：

```bash
python eval_report.py samples/eval_runs --out outputs/eval_report_sample
```

| 目录 | 内容 |
|---|---|
| matrix_vla | 原版 LIBERO 四个套件各 10 个任务的 VLA-only，控制步上限 300 |
| matrix_vla_pro | LIBERO-Pro 的 Position（_swap）和 Task（_task）扰动，各四个套件、各 10 个任务的 VLA-only |
| matrix_pair_v2 | 扰动套件上 Direct、Hybrid 的配对运行（Position 每套件 3 个任务；Task 只跑了 Direct） |
| orchestration | LIBERO-10 任务 0 的三个初态上，编排、写死顺序、VLA-only、Direct 四组，控制步上限 400 |

每批目录里的 declaration.json 是运行前冻结的评测声明，每条 episode 目录里有 report.json 和 ledger.sqlite。
report.json 可以删掉后用账本重算（见 harness/report.py），图片和逐段原始返回没有放进来。
编排那一轮是旧版预算规则：VLA-only 组的策略调用上限等于模型调用上限 30，也就是最多 300 步，三条都在 300 步内成功，结果不受影响。
