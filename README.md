# Embodied Harness

基于 LIBERO 的大模型与 VLA 协同执行及评测项目。

## 主要功能

- Direct、VLA-only 与 Hybrid 控制方式
- 动作校验、观测新鲜度检查与请求去重
- 执行账本与结构化回执
- 位置扰动评测与双目标任务编排

## 目录结构

- harness/：执行与评测代码
- first_loop/：仿真观测与基础动作示例
- toy_loop/：最小工具调用与回执示例
- check_libero.py：环境检查脚本

## 依赖说明

LIBERO 与 OpenPI 需独立安装。
模型权重、完整数据集及 API 密钥不包含在仓库中。
具体安装版本与运行命令待补充。
