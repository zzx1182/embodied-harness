# toy_loop · 「具身 Agent 的工具调用与 Harness」附件

几十行的状态机，把模型、runtime 和执行器的边界跑一遍。只用 Python 标准库，Windows、macOS、Linux 都能跑，不需要装任何东西。

```bash
python toy_loop.py
```

产物在 outputs/：proposal.json（原始建议）、events.json（校验与状态事件）、receipt.json（执行回执）。

可以改着玩的地方：

- fake_model 里把 delta_m 改成 [0.08, 0, 0]：每条建议都在校验这一步被拒（单次位移超过 0.05 m），执行次数是 0，故障 3 的演示到校验就结束。
- 故障 3 里把 timeout_s 改成 3.0：执行函数来得及返回，状态变成 COMPLETED，随后那一条也执行，执行次数 2。
- 只把 submit 里 request_id 已存在的那段检查注释掉，再跑故障 1：执行次数仍是 1，拦住重发的是观测序号（重发的请求带着旧的 obs_seq）。再把观测序号那段也注释掉，执行次数才变成 2。
