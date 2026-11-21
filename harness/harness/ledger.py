"""SQLite 账本。requests 一条逻辑请求一行，request_events 只追加，session_meta 一行，model_calls 每次调用一行。"""
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY, episode_id TEXT, executor_generation INTEGER, obs_seq INTEGER,
  payload_hash TEXT, payload_json TEXT, status TEXT, steps_requested INTEGER DEFAULT 0,
  steps_executed INTEGER DEFAULT 0, eef_before_json TEXT, eef_after_json TEXT, obs_seq_after INTEGER,
  native_success INTEGER, rejection_json TEXT, created_ts REAL, updated_ts REAL);
CREATE TABLE IF NOT EXISTS request_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, event TEXT, detail_json TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS session_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS scoring (
  id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT, obs_seq INTEGER, control_step INTEGER, progress_json TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS declarations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT, obs_seq INTEGER, control_step INTEGER, kind TEXT, subgoal TEXT,
  truth_json TEXT, image_path TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS policy_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT, obs_seq INTEGER, elapsed_s REAL, horizon INTEGER, ts REAL);
CREATE TABLE IF NOT EXISTS model_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT, obs_seq INTEGER, elapsed_s REAL,
  input_tokens INTEGER, output_tokens INTEGER, cached_input_tokens INTEGER, status TEXT, raw_path TEXT, ts REAL);
"""


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---- session ----
    def set_meta(self, **kv):
        for k, v in kv.items():
            self.db.execute("INSERT OR REPLACE INTO session_meta(key,value) VALUES(?,?)", (k, json.dumps(v)))

    def meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM session_meta WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    # ---- requests ----
    def get(self, request_id):
        r = self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
        return dict(r) if r else None

    def event(self, request_id, event, **detail):
        self.db.execute("INSERT INTO request_events(request_id,event,detail_json,ts) VALUES(?,?,?,?)",
                        (request_id, event, json.dumps(detail, ensure_ascii=False), time.time()))

    def accept(self, req, steps_requested, eef_before):
        """状态 ACCEPTED 落盘之后才允许步进仿真。"""
        now = time.time()
        self.db.execute(
            "INSERT INTO requests(request_id,episode_id,executor_generation,obs_seq,payload_hash,payload_json,status,"
            "steps_requested,eef_before_json,created_ts,updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (req.request_id, req.episode_id, req.executor_generation, req.obs_seq, req.payload_hash,
             json.dumps(req.payload(), ensure_ascii=False), "ACCEPTED", steps_requested,
             json.dumps(eef_before), now, now))
        self.event(req.request_id, "ACCEPTED")

    def running(self, request_id):
        self.db.execute("UPDATE requests SET status='RUNNING', updated_ts=? WHERE request_id=?", (time.time(), request_id))
        self.event(request_id, "RUNNING")

    def progress(self, request_id, steps_executed):
        """每确认一步写一次，崩溃后这就是已确认执行步数的下界。"""
        self.db.execute("UPDATE requests SET steps_executed=?, updated_ts=? WHERE request_id=?", (steps_executed, time.time(), request_id))

    def complete(self, request_id, steps_executed, eef_after, obs_seq_after, native_success, status="COMPLETED"):
        self.db.execute(
            "UPDATE requests SET status=?, steps_executed=?, eef_after_json=?, obs_seq_after=?, native_success=?, updated_ts=? WHERE request_id=?",
            (status, steps_executed, json.dumps(eef_after), obs_seq_after, int(bool(native_success)), time.time(), request_id))
        self.event(request_id, status, steps_executed=steps_executed)

    def reject(self, req, rejection, eef_before=None):
        now = time.time()
        self.db.execute(
            "INSERT OR IGNORE INTO requests(request_id,episode_id,executor_generation,obs_seq,payload_hash,payload_json,status,"
            "eef_before_json,eef_after_json,rejection_json,created_ts,updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (req.request_id, req.episode_id, req.executor_generation, req.obs_seq, req.payload_hash,
             json.dumps(req.payload(), ensure_ascii=False), "REJECTED", json.dumps(eef_before), json.dumps(eef_before),
             json.dumps(rejection, ensure_ascii=False), now, now))
        self.event(req.request_id, "REJECTED", **rejection)

    def mark_unknown_unfinished(self):
        """重启后扫描：没有终态的请求一律标 UNKNOWN，不补步数。"""
        rows = self.db.execute("SELECT request_id FROM requests WHERE status IN ('ACCEPTED','RUNNING')").fetchall()
        for r in rows:
            self.db.execute("UPDATE requests SET status='UNKNOWN', updated_ts=? WHERE request_id=?", (time.time(), r["request_id"]))
            self.event(r["request_id"], "UNKNOWN", reason="executor restarted before terminal state")
        return [r["request_id"] for r in rows]

    # ---- model calls ----
    def model_call(self, episode_id, obs_seq, elapsed_s, usage, status, raw_path):
        u = usage or {}
        self.db.execute(
            "INSERT INTO model_calls(episode_id,obs_seq,elapsed_s,input_tokens,output_tokens,cached_input_tokens,status,raw_path,ts) VALUES(?,?,?,?,?,?,?,?,?)",
            (episode_id, obs_seq, elapsed_s, u.get("input_tokens"), u.get("output_tokens"), u.get("cached_input_tokens"),
             status, str(raw_path) if raw_path else None, time.time()))

    # ---- scoring only: truth never flows back to the runtime ----
    def score(self, episode_id, obs_seq, control_step, progress):
        self.db.execute("INSERT INTO scoring(episode_id,obs_seq,control_step,progress_json,ts) VALUES(?,?,?,?,?)",
                        (episode_id, obs_seq, control_step, json.dumps(progress), time.time()))

    def scores(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM scoring ORDER BY id").fetchall()]

    def declare(self, episode_id, obs_seq, control_step, kind, subgoal, truth, image_path=None):
        self.db.execute("INSERT INTO declarations(episode_id,obs_seq,control_step,kind,subgoal,truth_json,image_path,ts) VALUES(?,?,?,?,?,?,?,?)",
                        (episode_id, obs_seq, control_step, kind, subgoal, json.dumps(truth), image_path, time.time()))

    def declarations(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM declarations ORDER BY id").fetchall()]

    def policy_call(self, episode_id, obs_seq, elapsed_s, horizon):
        self.db.execute("INSERT INTO policy_calls(episode_id,obs_seq,elapsed_s,horizon,ts) VALUES(?,?,?,?,?)",
                        (episode_id, obs_seq, elapsed_s, horizon, time.time()))

    def policy_calls(self, episode_id=None):
        q = "SELECT * FROM policy_calls" + (" WHERE episode_id=?" if episode_id else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (episode_id,) if episode_id else ()).fetchall()]

    # ---- reads for report ----
    def requests(self, episode_id=None):
        q = "SELECT * FROM requests" + (" WHERE episode_id=?" if episode_id else "") + " ORDER BY created_ts, request_id"
        return [dict(r) for r in self.db.execute(q, (episode_id,) if episode_id else ()).fetchall()]

    def events(self, request_id=None):
        q = "SELECT * FROM request_events" + (" WHERE request_id=?" if request_id else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (request_id,) if request_id else ()).fetchall()]

    def model_calls(self, episode_id=None):
        q = "SELECT * FROM model_calls" + (" WHERE episode_id=?" if episode_id else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (episode_id,) if episode_id else ()).fetchall()]

    def close(self):
        self.db.close()
