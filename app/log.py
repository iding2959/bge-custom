"""
日志配置模块 — 基于 sololib Loguru 实现
控制台输出 level，文件记录 file_level（含所有细节）

使用 logger.bind(client_id="xxx") 后 client_id 自动出现在 pid/tid 同列。
"""
from __future__ import annotations

from typing import Optional

from sololib.utils.logru_util import setup_logger as _setup_logger
from sololib.utils.logru_util import logger
from sololib.utils.logru_util import shutdown_logger

__all__ = ["setup_logging", "get_logger", "logger", "shutdown_logger"]


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    env: Optional[str] = None,
    enable_json: Optional[bool] = None,
    add_error_file: Optional[bool] = None,
    catch_unhandled: Optional[bool] = None,
    file_level: Optional[str] = None,
) -> None:
    """
    初始化日志系统（适配 sololib Loguru）

    控制台输出 level 级别（默认 INFO），文件记录 file_level 级别（默认 DEBUG）。

    :param level: 控制台日志级别（"DEBUG", "INFO", "WARNING", "ERROR"），默认 INFO
    :param file_level: 文件日志级别，默认 "DEBUG"（文件记录全部细节）
    :param log_dir: 日志文件目录，默认 "logs"
    :param env: 环境类型 "dev" 或 "prod"，默认从 LOG_ENV 环境变量读取
    :param enable_json: 是否输出 JSON 结构化日志，默认 False
    :param add_error_file: 是否额外输出 error.log，默认 True
    :param catch_unhandled: 是否捕获未处理异常，默认 True
    """
    kwargs = dict(
        env=env,
        log_dir=log_dir,
        level=level,
        file_level=file_level,
        enable_json=enable_json,
        add_error_file=add_error_file,
        catch_unhandled=catch_unhandled,
        inline_fields={"client_id"},
    )
    # 过滤 explicit None 值，让 sololib 使用默认值
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    _setup_logger(**kwargs)


def get_logger(name: str = __name__):
    """
    获取 logger（返回 loguru 全局 logger）

    :param name: 模块名（loguru 自动从调用栈捕获模块/行号，该参数仅保留兼容）
    """
    return logger
