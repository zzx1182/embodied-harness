"""具身 Agent harness 最小实现，配合「模型建议、执行校验与可靠回执」使用。

模块分工和课文一致：
  contracts   四个契约的数据结构
  validator   对建议做四道校验
  ledger      SQLite 账本，执行器是唯一写入者
  executor    单写执行器，去重、观测新鲜度、UNKNOWN
  runtime     打包上下文、调模型、分配身份、预算、终止
  models      模型客户端（OpenAI 兼容接口、Codex CLI、脚本模型）
  envs        LIBERO 环境适配器和测试用的假环境
  report      从账本重算 episode 报告
"""
