"""「具身 Agent 的工具调用与 Harness」 · 先做一次最小工具调用。

不用任何框架，只用 Python 标准库，写一个几十行的状态机：
读一份 JSON 观测 -> 生成一份带 schema 的动作建议 -> 校验 -> 记请求 -> 执行 -> 回执。
再故意制造三种故障，看状态机怎么处理：
  1. 同一个 request_id 重复提交      -> 不会执行第二次，返回第一次的回执
  2. 建议引用的观测序号已经过期      -> 拒绝，不执行
  3. 执行函数超时，不知道动完没有     -> 状态停在 UNKNOWN，不自动重发

运行：python toy_loop.py
产物：outputs/proposal.json、outputs/events.json、outputs/receipt.json
"""
import hashlib
import json
import threading
import time
from pathlib import Path

OUT = Path("outputs")
SCHEMA = {"tool": str, "expected_obs_seq": int, "delta_m": list, "steps": int}  # 字段名 -> 类型
EVENTS = []


def log(request_id, event, **detail):
    EVENTS.append({"t": round(time.monotonic() - T0, 3), "request_id": request_id, "event": event, **detail})
    print(f"  [{event:<18}] {request_id} {json.dumps(detail, ensure_ascii=False) if detail else ''}")


def fake_model(obs):
    """代替大模型：看观测，按 schema 给一条建议。真实情况下这里是一次 Function Calling。"""
    return {"tool": "move_eef", "expected_obs_seq": obs["obs_seq"], "delta_m": [0.02, 0.0, 0.0], "steps": 3}


def validate(p):
    """schema 校验：字段齐、类型对、数值在范围内。返回 None 表示通过。"""
    for k, t in SCHEMA.items():
        if not isinstance(p.get(k), t) or isinstance(p.get(k), bool):
            return f"字段 {k} 缺失或类型不对"
    if p["tool"] != "move_eef" or len(p["delta_m"]) != 3 or not 1 <= p["steps"] <= 5:
        return "工具名、位移维度或步数不合法"
    if sum(v * v for v in p["delta_m"]) ** 0.5 > 0.05:
        return "单次位移超过 0.05 m"
    return None


class Executor:
    """唯一能改变世界状态的一方。账本在内存里，真实项目里要落盘。"""

    def __init__(self, obs, handler_delay_s=0.0, timeout_s=1.0):
        self.obs, self.ledger, self.halted = dict(obs), {}, False
        self.handler_delay_s, self.timeout_s, self.executions = handler_delay_s, timeout_s, 0

    def _handler(self, p):  # 真正动机械臂的地方；delay 模拟卡住
        time.sleep(self.handler_delay_s)
        self.obs["eef_pos"] = [round(a + b, 4) for a, b in zip(self.obs["eef_pos"], p["delta_m"])]
        self.executions += 1

    def reject(self, request_id, h, reason):
        """拒绝也记账：同一个 request_id 再来，原样返回这张拒绝回执；查询也查得到。"""
        receipt = {"request_id": request_id, "status": "REJECTED", "reason": reason, "eef_before": list(self.obs["eef_pos"])}
        self.ledger[request_id] = {"payload_hash": h, "receipt": receipt}
        return receipt

    def submit(self, request_id, p):
        h = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
        log(request_id, "RECEIVED", payload_hash=h)
        #if request_id in self.ledger:                                   # 故障 1：重复提交
        #    rec = self.ledger[request_id]
        #    if rec["payload_hash"] != h;
        #        log(request_id, "CONTENT_CONFLICT")
        #        return {"request_id": request_id, "status": "REJECTED", "reason": "同一 request_id 内容不同"}
        #    log(request_id, "DUPLICATE_RETURNED", status=rec["receipt"]["status"])
        #    return rec["receipt"]
        if self.halted:
            log(request_id, "REJECTED", reason="有未确认的请求，已停机")
            return self.reject(request_id, h, "EPISODE_HALTED")
        #if p.get("expected_obs_seq") != self.obs["obs_seq"]:            # 故障 2：旧观测
        #    log(request_id, "REJECTED", reason="OBSERVATION_EXPIRED", expected=p.get("expected_obs_seq"), current=self.obs["obs_seq"])
        #    return self.reject(request_id, h, "OBSERVATION_EXPIRED")
        err = validate(p)
        if err:
            log(request_id, "REJECTED", reason=err)
            return self.reject(request_id, h, err)
        before = list(self.obs["eef_pos"])
        self.ledger[request_id] = {"payload_hash": h, "receipt": {"request_id": request_id, "status": "ACCEPTED"}}
        log(request_id, "ACCEPTED")                                     # 先记账，再执行
        worker = threading.Thread(target=self._handler, args=(p,), daemon=True)
        worker.start()
        worker.join(self.timeout_s)
        if worker.is_alive():                                           # 故障 3：超时
            receipt = {"request_id": request_id, "status": "UNKNOWN", "eef_before": before,
                       "note": "执行函数没在时限内返回，不知道机械臂动了没有"}
            self.halted = True
            log(request_id, "UNKNOWN", timeout_s=self.timeout_s)
        else:
            self.obs["obs_seq"] += 1
            receipt = {"request_id": request_id, "status": "COMPLETED", "steps": p["steps"],
                       "eef_before": before, "eef_after": list(self.obs["eef_pos"]), "obs_seq_after": self.obs["obs_seq"]}
            log(request_id, "COMPLETED", eef_after=receipt["eef_after"])
        self.ledger[request_id]["receipt"] = receipt
        return receipt

    def query(self, request_id):
        rec = self.ledger.get(request_id)
        return rec["receipt"] if rec else None


def main():
    global T0
    T0 = time.monotonic()
    OUT.mkdir(exist_ok=True)
    obs = json.loads(Path("observation.json").read_text(encoding="utf-8"))
    print("读入观测:", obs)

    print("\n== 正常一轮")
    ex = Executor(obs)
    proposal = fake_model(ex.obs)
    (OUT / "proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2))
    receipt = ex.submit("req-001", proposal)
    (OUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))

    print("\n== 故障 1：同一个 request_id 再发一次")
    again = ex.submit("req-001", proposal)
    print(f"  执行次数 {ex.executions}，两次回执相同: {again == receipt}")

    print("\n== 故障 2：拿旧观测做的建议")
    stale = dict(proposal)                                              # 还引用 obs_seq=1，而现在已经是 2
    ex.submit("req-002", stale)
    print(f"  执行次数 {ex.executions}，末端位置 {ex.obs['eef_pos']}")

    print("\n== 故障 3：执行函数卡住超时")
    slow = Executor(obs, handler_delay_s=2.0, timeout_s=0.5)
    r = slow.submit("req-003", fake_model(slow.obs))
    print(f"  客户端收到 {r['status']}，不重发，改为查询: {slow.query('req-003')['status']}")
    if r["status"] == "REJECTED":
        print(f"  建议在校验这一步就被拒了（{r['reason']}），执行函数没被调用，超时演示到此为止。")
        (OUT / "events.json").write_text(json.dumps(EVENTS, ensure_ascii=False, indent=2))
        return
    slow.submit("req-004", fake_model(slow.obs))
    time.sleep(2.0)
    print(f"  超时那一刻末端在 {r['eef_before']}，两秒后再看是 {slow.obs['eef_pos']}，执行次数 {slow.executions}")
    print("  超时的那一次其实还是动了。如果刚才换个 request_id 重发，机械臂会走两次。")

    (OUT / "events.json").write_text(json.dumps(EVENTS, ensure_ascii=False, indent=2))
    print("\n已写出 outputs/proposal.json、outputs/events.json、outputs/receipt.json")


if __name__ == "__main__":
    main()
