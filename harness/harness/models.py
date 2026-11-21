"""模型客户端。三种：OpenAI 兼容接口、Codex CLI、脚本模型。都返回 (决策 dict, usage dict, 耗时秒)。"""
import base64
import io
import json
import os
import hashlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You are the decision model inside a LIBERO tabletop robot simulation. You control a Franka arm only through two tools and must answer with ONE JSON object and nothing else.
Inputs each turn: two 256x256 camera images (front view first, wrist camera second), the task instruction, five proprioception fields (joint positions, joint velocities, end-effector position in meters, end-effector quaternion in xyzw order, gripper opening), the current obs_seq, and the public summary of the previous execution receipt. No object poses and no success flag are ever given; the environment judges success by itself.
Tool move_eef: {"tool":"move_eef","expected_obs_seq":<int>,"target":{"frame_id":"robot0_base","position_m":[x,y,z],"quaternion_xyzw":[x,y,z,w],"gripper":"open"|"close"},"horizon_steps":<1..5>,"reason":"<short>"}
  position_m is an ABSOLUTE target in the same frame as the reported end-effector position; it must lie within 0.05 m of the current end-effector position. quaternion_xyzw is an absolute orientation within 0.35 rad of the current one (copy the current quaternion to keep orientation). The executor tracks the target for horizon_steps ticks at 20 Hz and reports the measured result; small moves are often not fully reached, read eef_after in the receipt instead of assuming.
Tool give_up: {"tool":"give_up","expected_obs_seq":<int>,"reason_category":"target_not_visible"|"stuck"|"unsafe"|"other","reason":"<short>"}
Hard limit: the Euclidean distance between position_m and the CURRENT robot0_eef_pos must be at most 0.05 m, so keep each target within about 0.04 m of the current position and plan many small segments. If the previous proposal was REJECTED (see last_receipt/history), the rejection lists code, the value you gave (got) and the limit; fix exactly that and propose again on the same obs_seq, do not repeat the rejected target.
Always copy the current obs_seq into expected_obs_seq. Never claim success. Plan in small steps: approach above the object, lower, close the gripper, lift, move over the goal, lower, open."""

HYBRID_ADDENDUM = """
Hybrid mode: the input also contains candidate_action_prefix, the first rows of an action chunk predicted by a learned VLA policy from the same observation. Each row is a native normalized controller command [dx,dy,dz,rx,ry,rz,gripper] in [-1,1] (translation scaled by 0.05 m, rotation by 0.5 rad per tick, gripper -1 open / +1 close). You may reply with a third tool:
{"tool":"accept_vla","expected_obs_seq":<int>,"horizon_steps":<1..max_prefix_steps>,"reason":"<short>"}
which executes that many rows unchanged; the unexecuted suffix is discarded. Use accept_vla when the candidate looks useful, otherwise issue a small move_eef correction or give_up."""

ORCHESTRATOR_PROMPT = """You orchestrate a learned manipulation policy (pi0.5, trained on LIBERO tasks) in a tabletop robot simulation. You never output motor commands; every arm motion comes from the policy. Answer with ONE JSON object and nothing else.
Inputs each turn: two 256x256 camera images (front view first, wrist camera second), proprioception, the overall task instruction, the current obs_seq, the subgoals you already declared, and a short history of the skills you ran.
Tools (fill unused fields with "" or 0):
run_skill: {"tool":"run_skill","expected_obs_seq":<int>,"instruction":"<one short single-object instruction in the style of the policy's training data, for example 'pick up the alphabet soup and place it in the basket'>","chunks":<1..max_skill_chunks>,"subgoal":"","reason":"<short>"}
  The policy runs `chunks` action chunks of 10 control steps each (0.5 s per chunk) with your instruction, then you see the new images.
declare_subgoal_done: {"tool":"declare_subgoal_done","expected_obs_seq":<int>,"instruction":"","chunks":0,"subgoal":"<what is now done, for example 'alphabet soup in basket'>","reason":"<short>"}
  Declare a subgoal only when you can see in the images that it is done. Declarations are recorded and audited later against the simulator.
finish: {"tool":"finish","expected_obs_seq":<int>,"instruction":"","chunks":0,"subgoal":"","reason":"<short>"} when you believe the whole task is complete. The environment judges success independently; finishing early ends the episode.
give_up: {"tool":"give_up","expected_obs_seq":<int>,"instruction":"","chunks":0,"subgoal":"","reason":"<short>"}
Always copy the current obs_seq into expected_obs_seq. Decompose the task into single-object sub-instructions and run them one at a time, checking the images after each run."""


def system_prompt(context):
    if context.get("mode") == "orchestrate":
        return ORCHESTRATOR_PROMPT
    return SYSTEM_PROMPT + (HYBRID_ADDENDUM if "candidate_action_prefix" in context else "")



def _png_bytes(array):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _usage_from_openai(u):
    if not u:
        return None
    return {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
            "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}


# 这几种 HTTP 错误说明配置写错了（密钥、模型名、请求参数），不属于接口繁忙
FATAL_HTTP = (400, 401, 403, 404)
# Codex CLI 报这些说明登录或额度出了问题，等多久都不会好
CODEX_FATAL = ("usageLimitExceeded", "usage limit", "401 Unauthorized", "not logged in", "model_not_found")


class OpenAICompatibleModel:
    """任何 OpenAI 兼容的 chat/completions 接口。通过环境变量配置。"""

    def __init__(self, base_url=None, model=None, api_key=None, timeout_s=120, temperature=0.0):
        self.base_url = (base_url or os.environ.get("HARNESS_MODEL_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("HARNESS_MODEL_NAME", "")
        self.api_key = api_key or os.environ.get("HARNESS_MODEL_API_KEY", "")
        self.timeout_s = timeout_s
        self.temperature = temperature
        if not self.model:
            raise SystemExit("请设置 HARNESS_MODEL_NAME（以及 HARNESS_MODEL_BASE_URL、HARNESS_MODEL_API_KEY）")
        self.identity = {"kind": "openai_compatible", "base_url": self.base_url, "model": self.model}

    def decide(self, context, images, out_dir=None):
        content = [{"type": "text", "text": "INPUT\n" + json.dumps(context, ensure_ascii=False)}]
        for name, arr in images.items():
            b64 = base64.b64encode(_png_bytes(arr)).decode()
            content.append({"type": "text", "text": f"image: {name}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system_prompt(context)}, {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode(errors="replace")[:300]
            except Exception:
                detail = ""
            msg = f"HTTP {e.code}: {detail or e.reason}"
            if e.code in FATAL_HTTP:  # 密钥、模型名、参数写错，重试多少次都一样，直接停机报错
                return {"_fatal": msg}, None, time.monotonic() - started
            return {"_error": msg}, None, time.monotonic() - started   # 429、5xx 退避重试
        except Exception as e:  # 超时、网络断开算传输失败，和解析失败分开记
            return {"_error": f"{type(e).__name__}: {str(e)[:300]}"}, None, time.monotonic() - started
        elapsed = time.monotonic() - started
        if out_dir:
            (out_dir / "model_raw.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
        text = data["choices"][0]["message"]["content"]
        try:
            decision = json.loads(text)
        except json.JSONDecodeError:
            decision = {"_parse_error": text[:500]}
        return decision, _usage_from_openai(data.get("usage")), elapsed


class CodexCLIModel:
    """作者环境用的 Codex CLI（gpt-6-astra）。学员用 OpenAI 兼容接口即可，这个类只为复现作者记录。"""

    def __init__(self, codex_bin=None, model=None, run_as_user=None, scratch=None, reasoning="medium", timeout_s=240):
        self.codex_bin = codex_bin or os.environ.get("HARNESS_CODEX_BIN", "codex")
        self.model = model or os.environ.get("HARNESS_CODEX_MODEL", "gpt-6-astra")
        self.run_as_user = run_as_user or os.environ.get("HARNESS_CODEX_USER")
        # Codex 的工作目录每个进程单独建一个，多人共用一台机器、或者同时跑几批时互不覆盖
        self.scratch = scratch or os.environ.get("HARNESS_CODEX_SCRATCH") or tempfile.mkdtemp(prefix="harness-codex-")
        self.reasoning = reasoning
        self.timeout_s = timeout_s
        self.identity = {"kind": "codex_cli", "model": self.model, "reasoning_effort": reasoning}
        self._schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "tool": {"type": "string", "enum": ["move_eef", "give_up", "accept_vla"]},
                "expected_obs_seq": {"type": "integer"},
                "target": {"type": "object", "additionalProperties": False, "properties": {
                    "frame_id": {"type": "string"}, "position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                    "quaternion_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "gripper": {"type": "string", "enum": ["open", "close"]}},
                    "required": ["frame_id", "position_m", "quaternion_xyzw", "gripper"]},
                "horizon_steps": {"type": "integer"}, "reason": {"type": "string"},
                "reason_category": {"type": "string", "enum": ["target_not_visible", "stuck", "unsafe", "other", "none"]}},
            "required": ["tool", "expected_obs_seq", "target", "horizon_steps", "reason", "reason_category"]}
        self._orchestrate_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "tool": {"type": "string", "enum": ["run_skill", "declare_subgoal_done", "finish", "give_up"]},
                "expected_obs_seq": {"type": "integer"}, "instruction": {"type": "string"},
                "chunks": {"type": "integer"}, "subgoal": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["tool", "expected_obs_seq", "instruction", "chunks", "subgoal", "reason"]}

    def _wrap(self, cmd):
        return (["runuser", "-u", self.run_as_user, "--"] if self.run_as_user else []) + cmd

    def decide(self, context, images, out_dir=None):
        from pathlib import Path
        from PIL import Image
        out_dir = Path(out_dir or self.scratch)
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(self._wrap(["mkdir", "-p", self.scratch]), check=True)
        schema_path = out_dir / "schema.json"
        schema_path.write_text(json.dumps(self._orchestrate_schema if context.get("mode") == "orchestrate" else self._schema))
        image_paths = []
        for name, arr in images.items():
            p = out_dir / f"{name}.png"
            Image.fromarray(arr).save(p)
            image_paths.append(p)
        system = system_prompt(context)
        prompt = system + "\nINPUT\n" + json.dumps(context, ensure_ascii=False)
        (out_dir / "prompt.txt").write_text(prompt)
        # 输出文件名按段目录的完整路径取摘要，不同 episode 的同号段不会撞名
        output = Path(self.scratch) / (hashlib.sha1(str(out_dir.resolve()).encode()).hexdigest()[:12] + "-decision.json")
        cmd = [self.codex_bin, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
               "--model", self.model, "-C", self.scratch, "-c", f'model_reasoning_effort="{self.reasoning}"', "-c", 'web_search="disabled"']
        for feature in ["shell_tool", "apps", "plugins", "multi_agent", "browser_use", "computer_use", "hooks", "image_generation"]:
            cmd += ["--disable", feature]
        cmd += ["--json", "--output-schema", str(schema_path), "-o", str(output)]
        for p in image_paths:
            cmd += ["-i", str(p)]
        cmd += ["-"]
        started = time.monotonic()
        try:
            result = subprocess.run(self._wrap(cmd), input=prompt, text=True, capture_output=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return {"_error": f"codex timeout after {self.timeout_s}s"}, None, time.monotonic() - started
        elapsed = time.monotonic() - started
        (out_dir / "events.jsonl").write_text(result.stdout)
        (out_dir / "stderr.log").write_text(result.stderr)
        usage, completed = None, False
        for line in result.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                completed = True
                u = ev.get("usage") or {}
                usage = {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                         "cached_input_tokens": u.get("cached_input_tokens")}
        if result.returncode or not completed or not output.exists():
            errors = []
            for l in result.stdout.splitlines():
                try:
                    ev = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") in ("error", "turn.failed"):
                    errors.append(ev.get("message") or (ev.get("error") or {}).get("message") or l[:200])
            msg = f"codex exit {result.returncode}: {errors[-1] if errors else result.stderr[-300:]}"
            if any(k.lower() in msg.lower() for k in CODEX_FATAL):
                return {"_fatal": msg}, usage, elapsed          # 登录失效、额度用完：停机，别空等
            return {"_error": msg}, usage, elapsed
        decision = json.loads(output.read_text())
        (out_dir / "model_raw.json").write_text(json.dumps(decision, indent=2))
        if context.get("mode") == "orchestrate":
            pass
        elif decision.get("tool") == "give_up":
            decision.pop("target", None)
            decision.pop("horizon_steps", None)
        elif decision.get("tool") == "accept_vla":
            decision.pop("target", None)
            decision.pop("reason_category", None)
        else:
            decision.pop("reason_category", None)
        return decision, usage, elapsed


class ScriptedModel:
    """按脚本返回预先写好的建议，回归用例用。每个条目可以是 dict，或接收 context 返回 dict 的函数。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.identity = {"kind": "scripted"}

    def decide(self, context, images, out_dir=None):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        decision = item(context) if callable(item) else dict(item)
        if "expected_obs_seq" not in decision:
            decision["expected_obs_seq"] = context["obs_seq"]
        return decision, {"input_tokens": 1000, "output_tokens": 50, "cached_input_tokens": 0}, 0.01


def make_model(kind, **kw):
    return {"openai": OpenAICompatibleModel, "codex": CodexCLIModel}[kind](**kw)


def model_identity(kind):
    """不建模型对象，只按环境变量算出模型身份，和默认参数下模型对象的 identity 相同。run_matrix 用它冻结进声明。"""
    if kind == "openai":
        return {"kind": "openai_compatible", "base_url": os.environ.get("HARNESS_MODEL_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                "model": os.environ.get("HARNESS_MODEL_NAME", "")}
    return {"kind": "codex_cli", "model": os.environ.get("HARNESS_CODEX_MODEL", "gpt-6-astra"), "reasoning_effort": "medium"}
