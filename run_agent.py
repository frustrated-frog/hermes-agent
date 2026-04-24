#!/usr/bin/env python3
"""
AI Agent Runner with Tool Calling - Hermes 核心智能体执行器

这是 Hermes 的核心模块，提供了一个完整的 AI 智能体运行环境，专门用于执行带有工具调用能力的 AI 模型。
该模块负责整个对话循环、工具执行和响应管理的完整生命周期。

核心架构原理：
- 基于 OpenAI API 兼容的客户端架构，支持多种模型提供商
- 实现了完整的工具调用循环：解析工具调用 → 执行工具 → 返回结果 → 继续对话
- 内置了复杂的并发控制和错误恢复机制
- 支持多轮对话历史管理和上下文压缩
- 集成了完整的系统提示词构建和模型元数据管理

关键特性：
- 自动工具调用循环直到任务完成：智能体可以连续调用多个工具来完成复杂任务
- 可配置的模型参数：支持温度、最大令牌数、上下文窗口等参数的动态调整
- 完整的错误处理和恢复机制：包含网络重试、模型降级、上下文溢出处理等
- 消息历史管理：智能维护对话历史，支持上下文压缩和记忆管理
- 多模型提供商支持：通过统一的 OpenAI 兼容接口支持 Claude、GPT 等多种模型

使用示例：
    from run_agent import AIAgent
    
    # 创建智能体实例，配置模型端点和具体模型
    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    
    # 运行对话，智能体会自动处理工具调用和响应
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

# 系统级和标准库导入 - 为智能体运行提供基础功能支持
import atexit  # 注册程序退出时的清理函数
import asyncio  # 异步编程支持，用于并发工具执行
import base64  # Base64 编码解码，用于处理二进制数据
import concurrent.futures  # 线程池和进程池，用于并行工具执行
import copy  # 深拷贝支持，用于创建对象的安全副本
import hashlib  # 哈希算法，用于生成唯一标识符和缓存键
import json  # JSON 序列化/反序列化，用于 API 通信和数据存储
import logging  # 日志系统，用于记录运行状态和调试信息
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例
import os  # 操作系统接口，用于环境变量和文件系统操作
import random  # 随机数生成，用于加载动画和随机选择
import re  # 正则表达式，用于文本模式匹配和验证
import sys  # 系统特定参数和函数，用于标准输入输出控制
import tempfile  # 临时文件和目录处理，用于安全文件操作
import time  # 时间相关函数，用于性能计时和延迟
import threading  # 线程支持，用于并发控制和线程安全
import weakref  # 弱引用，用于内存管理和避免循环引用
from types import SimpleNamespace  # 简单的属性容器，用于创建轻量级对象
import uuid  # 通用唯一标识符生成，用于会话和请求跟踪
from typing import List, Dict, Any, Optional  # 类型注解，提供代码静态类型检查
from openai import OpenAI  # OpenAI 客户端库，提供与 OpenAI API 兼容的接口
import fire  # 命令行接口生成库，用于创建 CLI 工具
from datetime import datetime  # 日期时间处理，用于时间戳和日志记录
from pathlib import Path  # 面向对象的路径操作，提供跨平台的文件路径处理

# Hermes 框架核心导入 - 提供框架级别的功能支持
from hermes_constants import get_hermes_home  # 获取 Hermes 主目录路径的常量函数

# 环境变量加载机制 - 确保用户配置优先于系统默认
# 加载顺序：~/.hermes/.env 优先，项目根目录的 .env 作为开发环境后备
# 这种设计确保用户管理的 env 文件可以覆盖过时的 shell 环境变量
from hermes_cli.env_loader import load_hermes_dotenv  # Hermes 专用的环境变量加载器

# 环境变量加载执行 - 构建完整的配置环境
_hermes_home = get_hermes_home()  # 获取 Hermes 主目录路径，通常是 ~/.hermes
_project_env = Path(__file__).parent / '.env'  # 项目根目录下的 .env 文件路径，用于开发环境
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)  # 执行环境变量加载

# 环境加载结果处理 - 提供清晰的配置反馈
if _loaded_env_paths:  # 如果成功加载了环境变量文件
    for _env_path in _loaded_env_paths:  # 遍历所有加载的文件路径
        logger.info("Loaded environment variables from %s", _env_path)  # 记录加载信息，便于调试
else:  # 如果没有找到环境变量文件
    logger.info("No .env file found. Using system environment variables.")  # 使用系统环境变量


# 工具系统导入 - 构建智能体的工具执行能力
# 这些模块提供了智能体与外部环境交互的核心功能
from model_tools import (
    get_tool_definitions,  # 获取所有可用工具的定义信息
    get_toolset_for_tool,  # 根据工具名称获取对应的工具集
    handle_function_call,  # 处理函数调用的核心逻辑
    check_toolset_requirements,  # 检查工具集的运行时要求
)
from tools.terminal_tool import cleanup_vm  # 虚拟机清理函数，确保环境安全
from tools.interrupt import set_interrupt as _set_interrupt  # 中断处理机制
from tools.browser_tool import cleanup_browser  # 浏览器资源清理函数


# 核心配置常量导入
from hermes_constants import OPENROUTER_BASE_URL  # OpenRouter API 的基础 URL

# 智能体内部模块导入 - 核心功能模块化设计
# 这些模块将智能体的复杂功能分解为独立的、可维护的组件
from agent.memory_manager import build_memory_context_block  # 构建记忆上下文块
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,  # 默认的智能体身份定义
    PLATFORM_HINTS,  # 平台相关的提示信息
    MEMORY_GUIDANCE,  # 记忆管理指导
    SESSION_SEARCH_GUIDANCE,  # 会话搜索功能指导
    SKILLS_GUIDANCE,  # 技能系统指导
    build_nous_subscription_prompt,  # 构建 Nous 订阅相关的提示
)
# 模型元数据管理 - 提供模型相关的信息和优化
from agent.model_metadata import (
    fetch_model_metadata,  # 获取模型元数据信息
    estimate_tokens_rough,  # 粗略估计令牌数量
    estimate_messages_tokens_rough,  # 估计消息令牌数量
    estimate_request_tokens_rough,  # 估计请求令牌数量
    get_next_probe_tier,  # 获取下一个探测层级
    parse_context_limit_from_error,  # 从错误中解析上下文限制
    save_context_length,  # 保存上下文长度信息
    is_local_endpoint,  # 判断是否为本地端点
)
from agent.context_compressor import ContextCompressor  # 上下文压缩器，用于处理长上下文
from agent.subdirectory_hints import SubdirectoryHintTracker  # 子目录提示跟踪器
from agent.prompt_caching import apply_anthropic_cache_control  # Anthropic 缓存控制
from agent.prompt_builder import (
    build_skills_system_prompt,  # 构建技能系统提示
    build_context_files_prompt,  # 构建上下文文件提示
    load_soul_md,  # 加载 SOUL.md 文件
    TOOL_USE_ENFORCEMENT_GUIDANCE,  # 工具使用强制指导
    TOOL_USE_ENFORCEMENT_MODELS,  # 工具使用强制模型列表
    DEVELOPER_ROLE_MODELS,  # 开发者角色模型列表
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,  # Google 模型操作指导
    OPENAI_MODEL_EXECUTION_GUIDANCE,  # OpenAI 模型执行指导
)
from agent.usage_pricing import estimate_usage_cost, normalize_usage  # 使用量计费和标准化
from agent.display import (
    KawaiiSpinner,  # 可爱的加载动画
    build_tool_preview as _build_tool_preview,  # 构建工具预览
    get_cute_tool_message as _get_cute_tool_message_impl,  # 获取可爱的工具消息
    _detect_tool_failure,  # 检测工具失败
    get_tool_emoji as _get_tool_emoji,  # 获取工具表情符号
)
from agent.trajectory import (
    convert_scratchpad_to_think,  # 将草稿转换为思考内容
    has_incomplete_scratchpad,  # 检查是否有不完整的草稿
    save_trajectory as _save_trajectory_to_file,  # 保存轨迹到文件
)
from utils import atomic_json_write, env_var_enabled  # 原子 JSON 写入和环境变量检查



# 安全输出包装器 - 防止管道错误导致智能体崩溃
class _SafeWriter:
    """
    透明的标准输入输出包装器，用于捕获管道损坏导致的 OSError/ValueError。
    
    设计原理：
    当 hermes-agent 作为 systemd 服务、Docker 容器或无头守护进程运行时，stdout/stderr
    管道可能因为空闲超时、缓冲区耗尽或套接字重置而变得不可用。此时任何 print() 调用都会引发
    OSError: [Errno 5] Input/output error，这可能导致智能体设置或 run_conversation() 崩溃，
    特别是当异常处理程序也尝试打印时，会通过双重故障导致崩溃。
    
    另外，当子智能体在 ThreadPoolExecutor 线程中运行时，共享的 stdout 句柄可能在线程
    拆解和清理之间关闭，引发 ValueError: I/O operation on closed file 而不是 OSError。
    
    这个包装器将所有写操作委托给底层流，并静默捕获 OSError 和 ValueError。当被包装的
    流健康时，它是完全透明的，不会影响正常操作。
    
    这种设计的核心价值在于：确保智能体即使在恶劣的运行环境下也能保持稳定运行，
    不会因为输出问题而意外终止，提供了强大的容错能力。
    """

    # 使用 __slots__ 优化内存使用，限制实例属性
    __slots__ = ("_inner",)

    def __init__(self, inner):
        # 使用 object.__setattr__ 避免递归调用，直接设置内部流对象
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        """安全写操作 - 捕获所有可能的 I/O 错误"""
        try:
            # 尝试将数据写入底层流
            return self._inner.write(data)
        except (OSError, ValueError):
            # 如果发生错误，返回数据长度假装写入成功
            # 这样调用者不会察觉到错误，保持接口一致性
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        """安全刷新操作 - 静默处理刷新错误"""
        try:
            # 尝试刷新底层流
            self._inner.flush()
        except (OSError, ValueError):
            # 静默捕获错误，不传播异常
            pass

    def fileno(self):
        """获取文件描述符 - 直接委托给底层流"""
        return self._inner.fileno()

    def isatty(self):
        """检查是否为终端 - 安全版本，捕获可能的错误"""
        try:
            # 尝试检查是否为 TTY
            return self._inner.isatty()
        except (OSError, ValueError):
            # 如果检查失败，假设不是 TTY
            return False

    def __getattr__(self, name):
        """动态属性访问 - 将所有其他属性委托给底层流"""
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """安装安全的标准输入输出包装器。
    
    这个函数在智能体启动时被调用，用于包装 stdout 和 stderr 流。
    它的核心作用是确保即使在最恶劣的运行环境下，智能体的输出操作也不会导致崩溃。
    
    设计考虑：
    - 守护进程、容器化环境中管道可能随时断开
    - 网络连接中断可能导致输出流不可用
    - 多线程环境下共享流的竞争条件
    
    通过包装这些流，我们为智能体提供了一个稳定的输出层，这是构建可靠 AI 系统的基础。
    """
    # 遍历标准输出和标准错误流
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)  # 获取流对象
        if stream is not None and not isinstance(stream, _SafeWriter):
            # 只有当流存在且未被包装时才进行包装
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """线程安全的迭代计数器 - 智能体的"燃料"管理系统。
    
    这个类实现了智能体执行的核心限制机制，防止智能体无限循环或过度消耗资源。
    它是构建可靠 AI 系统的关键组件，确保智能体在合理的时间和资源范围内完成任务。
    
    设计原理：
    - 每个智能体（父智能体或子智能体）都有自己的独立预算
    - 父智能体的预算上限为 max_iterations（默认 90 次迭代）
    - 每个子智能体有独立的预算上限为 delegation.max_iterations（默认 50 次迭代）
    - 这意味着父智能体 + 所有子智能体的总迭代次数可以超过父智能体的上限
    - 用户可以通过 config.yaml 中的 delegation.max_iterations 配置每个子智能体的限制
    
    特殊处理：
    execute_code（编程式工具调用）的迭代会通过 refund() 方法返还，
    这样编程任务不会消耗对话预算，确保技术任务可以充分执行而不影响对话质量。
    
    这种设计平衡了：
    - 防止无限循环：确保智能体不会陷入死循环
    - 资源控制：防止过度消耗 API 配额和计算资源
    - 任务完整性：允许复杂的多步骤任务完成
    - 子智能体独立性：每个委托的任务有自己的预算空间
    """

    def __init__(self, max_total: int):
        """初始化迭代预算。
        
        Args:
            max_total: 最大迭代次数，这是智能体可以执行的总步数限制
        """
        self.max_total = max_total  # 保存最大迭代次数
        self._used = 0  # 已使用的迭代次数，初始为 0
        self._lock = threading.Lock()  # 创建线程锁，确保多线程环境下的线程安全

    def consume(self) -> bool:
        """尝试消耗一次迭代。
        
        这是预算管理的核心方法，每次智能体要执行一步操作时都会调用。
        
        Returns:
            bool: 如果允许消耗（还有剩余预算）返回 True，否则返回 False
            
        线程安全考虑：
        使用 with self._lock 确保在多线程环境下，预算的检查和更新是原子操作，
        防止出现竞态条件导致预算超支。
        """
        with self._lock:  # 获取线程锁，确保操作的原子性
            if self._used >= self.max_total:  # 检查是否已经用完所有预算
                return False  # 预算已用完，不允许继续执行
            self._used += 1  # 消耗一次迭代
            return True  # 成功消耗，允许继续执行

    def refund(self) -> None:
        """返还一次迭代（例如用于 execute_code 回合）。
        
        这个方法用于特殊情况的预算管理，主要应用于编程相关的工具调用。
        原理是编程任务（如代码执行）通常是对话的一部分，但它们本身不应该
        消耗对话预算，否则可能导致真正有意义的对话步骤被限制。
        
        使用场景：
        - execute_code 工具调用后，返还消耗的迭代
        - 某些内部工具调用不应该是用户可见的对话步骤
        - 错误恢复时返还意外消耗的迭代
        
        这种设计确保了技术任务可以充分执行，同时保持对话质量。
        """
        with self._lock:  # 确保线程安全
            if self._used > 0:  # 只有在确实消耗过的情况下才返还
                self._used -= 1  # 返还一次迭代到预算池

    @property
    def used(self) -> int:
        """获取已使用的迭代次数。
        
        这是一个属性装饰器方法，允许像访问属性一样使用：budget.used
        由于 self._used 是简单的整数读取，不需要锁保护。
        """
        return self._used  # 直接返回已使用的迭代次数

    @property
    def remaining(self) -> int:
        """获取剩余的迭代次数。
        
        计算并返回还可以执行多少次迭代。
        由于涉及计算和可能的并发访问，需要线程锁保护。
        """
        with self._lock:  # 确保线程安全
            return max(0, self.max_total - self._used)  # 确保不会返回负数


# 工具并发执行策略 - 智能体的并行计算控制中心

# 绝对禁止并发执行的工具集合 - 这些工具涉及用户交互或状态一致性
# 当这些工具出现在批处理中时，系统会回退到顺序执行
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})  # clarify 工具需要用户交互，必须串行化

# 可以安全并行执行的只读工具 - 这些工具不修改共享状态
# 它们没有可变的会话状态，因此可以安全地并发执行
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",      # Home Assistant 获取状态 - 只读操作
    "ha_list_entities",  # Home Assistant 列出实体 - 只读操作
    "ha_list_services",  # Home Assistant 列出服务 - 只读操作
    "read_file",         # 文件读取 - 只读操作，无状态修改
    "search_files",      # 文件搜索 - 只读操作，不影响现有文件
    "session_search",    # 会话搜索 - 只读历史查询
    "skill_view",        # 技能查看 - 只读技能内容获取
    "skills_list",       # 技能列表 - 只读技能枚举
    "vision_analyze",    # 视觉分析 - 独立的图像处理
    "web_extract",       # 网页提取 - 只读网络数据获取
    "web_search",        # 网络搜索 - 只读信息检索
})

# 路径作用域工具 - 当目标路径独立时可以并发执行
# 这些工具的操作范围受文件路径限制，只要路径不冲突就可以并行
_PATH_SCOPED_TOOLS = frozenset({
    "read_file",   # 文件读取 - 可以并发读取不同文件
    "write_file",  # 文件写入 - 可以并发写入不同文件
    "patch"        # 文件修补 - 可以并发修改不同文件
})

# 并行工具执行的最大工作线程数
# 这个限制平衡了并发性能和系统资源消耗
# 设置过高可能导致系统资源耗尽，设置过低影响并行效率
_MAX_TOOL_WORKERS = 8

# 破坏性命令检测模式 - 安全保护机制
# 这些正则表达式用于识别可能修改或删除文件的终端命令

# 可能修改或删除文件的终端命令模式
# 这个正则表达式匹配常见的文件破坏性操作命令
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,  # 使用详细模式，允许注释和格式化
)

# 输出重定向覆盖模式 - 检测文件覆盖操作
# 匹配 > 重定向但不匹配 >> 追加（> 会覆盖文件内容）
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """启发式检测：这个终端命令是否看起来会修改/删除文件？
    
    这个函数是安全机制的一部分，用于在执行终端命令前评估其潜在风险。
    它不是绝对可靠的，但可以捕获大多数常见的破坏性操作模式。
    
    检测原理：
    1. 检查命令中是否包含已知的破坏性命令模式（rm, mv, sed -i 等）
    2. 检查是否使用了覆盖重定向（> 而不是 >>）
    
    这种启发式检测的价值在于：
    - 提前警告：在执行前识别潜在风险
    - 用户确认：对于破坏性操作要求额外确认
    - 审计记录：记录可能的破坏性操作用于安全审计
    
    Args:
        cmd: 要检测的终端命令字符串
        
    Returns:
        bool: 如果命令看起来具有破坏性返回 True，否则返回 False
    """
    # 基础安全检查 - 空命令直接返回安全
    if not cmd:  # 空命令不可能是破坏性的
        return False
    
    # 第一重检查：破坏性命令模式检测
    # 使用预编译的正则表达式快速识别常见的危险命令
    if _DESTRUCTIVE_PATTERNS.search(cmd):  # 检查破坏性命令模式
        return True
    
    # 第二重检查：输出重定向覆盖检测
    # 识别可能意外覆盖文件的重定向操作（> 但不包括 >>）
    if _REDIRECT_OVERWRITE.search(cmd):  # 检查覆盖重定向
        return True
    
    # 通过所有安全检查，命令看起来是安全的
    return False  # 没有发现破坏性特征


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    # 提取所有工具名称用于快速检查
    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):  # 检查是否有禁止并行的工具
        return False

    # 路径预留机制 - 用于检测文件操作的路径冲突
    reserved_paths: list[Path] = []
    
    # 逐个分析每个工具调用的并发安全性
    for tool_call in tool_calls:
        tool_name = tool_call.function.name  # 获取工具名称
        
        try:
            # 解析工具参数，用于进一步分析
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            # 参数解析失败，无法判断安全性，默认采用保守的串行执行
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):  # 参数必须是字典格式
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        # 路径作用域工具的特殊处理 - 检查路径冲突
        if tool_name in _PATH_SCOPED_TOOLS:
            # 提取工具的目标路径
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:  # 无法确定路径，采用保守策略
                return False
            # 检查是否与已预留的路径冲突
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False  # 路径冲突，不能并行
            # 预留这个路径，防止后续工具冲突
            reserved_paths.append(scoped_path)
            continue  # 这个工具可以并行，继续检查下一个

        # 检查工具是否在安全并行白名单中
        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False  # 不在白名单中的工具默认串行执行

    # 所有检查都通过，可以安全并行执行
    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve(); the file may not exist yet.
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # Empty paths shouldn't reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

_BUDGET_WARNING_RE = re.compile(
    r"\[BUDGET(?:\s+WARNING)?:\s+Iteration\s+\d+/\d+\..*?\]",
    re.DOTALL,
)


def _sanitize_surrogates(text: str) -> str:
    """Replace lone surrogate code points with U+FFFD (replacement character).

    Surrogates are invalid in UTF-8 and will crash ``json.dumps()`` inside the
    OpenAI SDK.  This is a fast no-op when the text contains no surrogates.
    """
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_messages_surrogates(messages: list) -> bool:
    """Sanitize surrogate characters from all string content in a messages list.

    Walks message dicts in-place.  Returns True if any surrogates were found
    and replaced, False otherwise.
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
    return found


def _strip_budget_warnings_from_history(messages: list) -> None:
    """从工具结果消息中移除预算压力警告 - 历史记录清理机制。
    
    预算警告是会话范围内的信号，绝不能泄露到重放的历史记录中。
    它们存在于工具结果 content 中，要么是 JSON 键（_budget_warning），
    要么是附加的纯文本。
    
    设计原理：
    - 预算警告是临时性的系统提示，用于提醒智能体注意迭代次数限制
    - 这些警告不应该成为永久对话历史的一部分，否则会污染真实的对话内容
    - 清理机制确保历史记录的纯净性，便于后续分析和重放
    
    实现方式：
    1. JSON 格式：检查并删除 _budget_warning 键
    2. 文本格式：使用正则表达式移除预算警告文本模式
    3. 原地修改：直接修改消息列表，避免创建新对象的开销
    
    这种清理的价值在于：
    - 数据纯净性：保持对话历史的真实性和完整性
    - 分析准确性：确保后续分析不会受到系统警告的干扰
    - 用户体验：避免预算相关的内部信息泄露给用户
    """
    # 遍历所有消息，查找需要清理的工具结果
    for msg in messages:
        # 只处理工具角色的消息，跳过其他类型的消息
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
            
        content = msg.get("content")
        # 快速检查：如果内容不包含预算警告相关的关键词，直接跳过
        if not isinstance(content, str) or "_budget_warning" not in content and "[BUDGET" not in content:
            continue

        # 首先尝试 JSON 格式清理（最常见的情况：_budget_warning 键在字典中）
        try:
            parsed = json.loads(content)  # 尝试解析为 JSON
            if isinstance(parsed, dict) and "_budget_warning" in parsed:
                # 找到预算警告键，删除它
                del parsed["_budget_warning"]
                # 重新序列化为 JSON 字符串
                msg["content"] = json.dumps(parsed, ensure_ascii=False)
                continue  # 成功处理，继续下一条消息
        except (json.JSONDecodeError, TypeError):
            # JSON 解析失败，回退到文本模式清理
            pass

        # 回退方案：使用正则表达式从纯文本工具结果中移除预算警告模式
        cleaned = _BUDGET_WARNING_RE.sub("", content).strip()
        if cleaned != content:
            # 如果清理后的内容与原始内容不同，说明移除了警告
            msg["content"] = cleaned


# =========================================================================
# 大工具结果处理器 — 将超大输出保存到临时文件
# =========================================================================

# 大结果处理的背景：
# 在 AI 智能体系统中，工具可能会返回非常大的结果（如日志文件、代码分析、大量数据等）。
# 如果将这些大结果直接包含在对话上下文中，会导致：
# - 上下文爆炸：迅速消耗可用的令牌配额
# - 性能下降：大文本的处理和传输会显著降低响应速度
# - 成本增加：更多的令牌意味着更高的 API 调用成本
# - 质量降低：大上下文可能稀释重要信息，影响模型理解

# 工具结果被保存到文件而不是内联保存的阈值。
# 100K 字符 ≈ 25K 令牌 — 对于任何合理的输出都很充裕，但能防止灾难性的上下文爆炸。
_LARGE_RESULT_CHARS = 100_000

# 原始结果中作为内联预览包含的字符数，
# 这样模型可以立即获得工具返回内容的上下文。
# 1500 字符足够提供有意义的预览，同时不会显著增加上下文负担。
_LARGE_RESULT_PREVIEW_CHARS = 1_500


def _save_oversized_tool_result(function_name: str, function_result: str) -> str:
    """将超大的工具结果替换为文件引用 + 预览。
    
    当工具返回的内容超过 _LARGE_RESULT_CHARS 字符时，完整内容会被写入
    HERMES_HOME/cache/tool_responses/ 下的临时文件，发送给模型的结果被替换为：
       • 一个简短的前端预览（前 _LARGE_RESULT_PREVIEW_CHARS 个字符）
       • 文件路径，以便模型可以使用 read_file / search_files 访问完整内容
    
    如果文件写入失败，则回退到破坏性截断。
    
    设计原理：
    - 透明性：对模型来说，这个过程应该是透明的，它仍然可以访问完整数据
    - 效率性：避免在 API 调用中传输大量文本数据
    - 可靠性：提供文件写入失败的回退机制
    - 可访问性：模型可以通过文件工具重新获取完整数据
    
    工作流程：
    1. 检查结果大小是否超过阈值
    2. 如果超过，创建临时文件保存完整内容
    3. 生成包含预览和文件路径的替代消息
    4. 如果文件写入失败，回退到截断模式
    
    Args:
        function_name: 工具函数的名称，用于生成文件名
        function_result: 工具的完整输出结果
        
    Returns:
        str: 处理后的结果，可能是原始结果、预览+文件路径、或截断版本
    """
    # 首先检查结果大小，如果未超过阈值直接返回原始结果
    original_len = len(function_result)
    if original_len <= _LARGE_RESULT_CHARS:
        # 结果在可接受范围内，无需特殊处理
        return function_result

    # 构建目标目录结构
    try:
        # 创建缓存目录：~/.hermes/cache/tool_responses/
        response_dir = os.path.join(get_hermes_home(), "cache", "tool_responses")
        os.makedirs(response_dir, exist_ok=True)  # 确保目录存在

        # 生成基于时间的唯一时间戳，精确到微秒
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 清理工具名称，确保文件名安全
        # 将非单词字符和非连字符替换为下划线，限制长度防止文件名过长
        safe_name = re.sub(r"[^\w\-]", "_", function_name)[:40]
        filename = f"{safe_name}_{timestamp}.txt"  # 构建唯一文件名
        filepath = os.path.join(response_dir, filename)  # 完整文件路径

        # 将完整结果写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(function_result)

        # 提取预览内容（前 N 个字符）
        preview = function_result[:_LARGE_RESULT_PREVIEW_CHARS]
        
        # 构建智能的替代消息，包含预览和文件访问信息
        return (
            f"{preview}\n\n"
            f"[Large tool response: {original_len:,} characters total — "
            f"only the first {_LARGE_RESULT_PREVIEW_CHARS:,} shown above. "
            f"Full output saved to: {filepath}\n"
            f"Use read_file or search_files on that path to access the rest.]"
        )
        
    except Exception as exc:
        # 文件写入失败时的回退策略：破坏性截断
        logger.warning("Failed to save large tool result to file: %s", exc)
        # 提供截断版本，并说明截断原因和失败信息
        return (
            function_result[:_LARGE_RESULT_CHARS]  # 截断到最大允许长度
            + f"\n\n[Truncated: tool response was {original_len:,} chars, "
            f"exceeding the {_LARGE_RESULT_CHARS:,} char limit. "
            f"File save failed: {exc}]"
        )


class AIAgent:
    """
    AI Agent with tool calling capabilities - 具有工具调用能力的 AI 智能体。
    
    这是 Hermes 的核心类，负责管理整个对话流程、工具执行和响应处理，
    专门为支持函数调用的 AI 模型设计。

    核心职责：
    - 对话管理：维护多轮对话历史，处理上下文压缩和记忆管理
    - 工具协调：解析工具调用请求，执行工具，处理返回结果
    - 模型交互：与各种 AI 模型提供商通信，处理不同的 API 格式
    - 错误恢复：处理网络错误、模型错误、工具执行错误等
    - 资源管理：管理迭代预算、上下文限制、并发执行等
    
    设计架构：
    - 模块化设计：将复杂功能分解为多个专门的组件
    - 可扩展性：支持多种模型提供商和工具集
    - 容错性：内置多重错误恢复和安全检查机制
    - 性能优化：支持并发执行、缓存、压缩等优化手段
    
    这个类是构建可靠、高效 AI 智能体系统的基础，它抽象了底层复杂性，
    为上层应用提供了简洁而强大的接口。
    """

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",  # 默认模型名称
        max_iterations: int = 90,  # 默认工具调用迭代次数（与子智能体共享）
        tool_delay: float = 1.0,  # 工具调用间隔延迟
        enabled_toolsets: List[str] = None,  # 启用的工具集白名单
        disabled_toolsets: List[str] = None,  # 禁用的工具集黑名单
        save_trajectories: bool = False,  # 是否保存对话轨迹到 JSONL 文件
        verbose_logging: bool = False,  # 是否启用详细日志用于调试
        quiet_mode: bool = False,  # 是否抑制进度输出以获得干净的 CLI 体验
        ephemeral_system_prompt: str = None,  # 执行期间使用的临时系统提示（不保存到轨迹）
        log_prefix_chars: int = 100,  # 日志预览显示的字符数
        log_prefix: str = "",  # 日志消息前缀，用于并行处理中的标识
        providers_allowed: List[str] = None,  # 允许的 OpenRouter 提供商
        providers_ignored: List[str] = None,  # 忽略的 OpenRouter 提供商
        providers_order: List[str] = None,  # OpenRouter 提供商尝试顺序
        provider_sort: str = None,  # 按价格/吞吐量/延迟排序提供商
        provider_require_parameters: bool = False,  # 是否要求提供商参数
        provider_data_collection: str = None,  # 提供商数据收集配置
        session_id: str = None,  # 会话 ID，用于日志记录
        tool_progress_callback: callable = None,  # 工具进度回调函数
        tool_start_callback: callable = None,  # 工具开始回调函数
        tool_complete_callback: callable = None,  # 工具完成回调函数
        thinking_callback: callable = None,  # 思考过程回调函数
        reasoning_callback: callable = None,  # 推理过程回调函数
        clarify_callback: callable = None,  # 澄清交互回调函数
        step_callback: callable = None,  # 步骤回调函数
        stream_delta_callback: callable = None,  # 流式增量回调函数
        tool_gen_callback: callable = None,  # 工具生成回调函数
        status_callback: callable = None,  # 状态回调函数
        max_tokens: int = None,  # 最大响应令牌数
        reasoning_config: Dict[str, Any] = None,  # 推理配置
        prefill_messages: List[Dict[str, Any]] = None,  # 预填充消息
        platform: str = None,  # 平台标识（cli、telegram、discord、whatsapp 等）
        skip_context_files: bool = False,  # 是否跳过上下文文件注入
        skip_memory: bool = False,  # 是否跳过记忆系统
        session_db=None,  # 会话数据库
        parent_session_id: str = None,  # 父会话 ID
        iteration_budget: "IterationBudget" = None,  # 迭代预算对象
        fallback_model: Dict[str, Any] = None,  # 回退模型配置
        credential_pool=None,  # 凭据池
        checkpoints_enabled: bool = False,  # 是否启用检查点
        checkpoint_max_snapshots: int = 50,  # 检查点最大快照数
        pass_session_id: bool = False,  # 是否传递会话 ID
        persist_session: bool = True,  # 是否持久化会话
    ):
        """
        初始化 AI 智能体 - 构建完整的智能体运行环境。
        
        这是 AIAgent 的核心构造函数，负责设置智能体运行所需的所有组件和配置。
        该方法处理了从模型配置到安全设置的完整初始化流程，是构建功能完整的
        AI 智能体的入口点。
        
        参数设计哲学：
        - 灵活性：支持多种模型提供商和配置选项
        - 可扩展性：通过回调函数支持自定义行为
        - 安全性：内置多重安全机制和限制
        - 性能优化：支持并发、缓存、压缩等优化手段
        
        关键配置类别：
        1. 模型配置：base_url、api_key、provider、model 等
        2. 行为控制：max_iterations、tool_delay、enabled_toolsets 等
        3. 用户体验：quiet_mode、verbose_logging、platform 等
        4. 回调系统：各种 callback 函数用于自定义行为
        5. 高级功能：reasoning_config、prefill_messages、checkpoints_enabled 等
        
        初始化流程：
        1. 安全标准输出安装
        2. 基础属性设置
        3. 模型客户端配置
        4. 日志系统设置
        5. 高级功能初始化
        6. 回调和状态管理设置
        
        Args:
            base_url (str): 模型 API 的基础 URL（可选）
            api_key (str): 身份验证的 API 密钥（可选，如果未提供则使用环境变量）
            provider (str): 提供商标识符（可选；用于遥测/路由提示）
            api_mode (str): API 模式覆盖："chat_completions" 或 "codex_responses"
            model (str): 要使用的模型名称（默认："anthropic/claude-opus-4.6"）
            max_iterations (int): 工具调用迭代的最大数量（默认：90）
            tool_delay (float): 工具调用之间的延迟（秒）（默认：1.0）
            enabled_toolsets (List[str]): 仅启用这些工具集中的工具（可选）
            disabled_toolsets (List[str]): 禁用这些工具集中的工具（可选）
            save_trajectories (bool): 是否将对话轨迹保存到 JSONL 文件（默认：False）
            verbose_logging (bool): 启用详细日志记录以进行调试（默认：False）
            quiet_mode (bool): 抑制进度输出以获得干净的 CLI 体验（默认：False）
            ephemeral_system_prompt (str): 智能体执行期间使用但不保存到轨迹的系统提示（可选）
            log_prefix_chars (int): 工具调用/响应的日志预览中显示的字符数（默认：100）
            log_prefix (str): 添加到所有日志消息的前缀，用于并行处理中的标识（默认：""）
            providers_allowed (List[str]): 允许的 OpenRouter 提供商（可选）
            providers_ignored (List[str]): 忽略的 OpenRouter 提供商（可选）
            providers_order (List[str]): 按顺序尝试的 OpenRouter 提供商（可选）
            provider_sort (str): 按价格/吞吐量/延迟排序提供商（可选）
            session_id (str): 用于日志记录的预生成会话 ID（可选，如果未提供则自动生成）
            tool_progress_callback (callable): 进度通知的回调函数(tool_name, args_preview)
            clarify_callback (callable): 交互式用户问题的回调函数(question, choices) -> str。
                由平台层（CLI 或网关）提供。如果为 None，澄清工具返回错误。
            max_tokens (int): 模型响应的最大令牌数（可选，如果未设置则使用模型默认值）
            reasoning_config (Dict): OpenRouter 推理配置覆盖（例如 {"effort": "none"} 禁用思考）。
                如果为 None，默认为 {"enabled": True, "effort": "medium"} 用于 OpenRouter。设置以禁用/自定义推理。
            prefill_messages (List[Dict]): 作为预填充上下文添加到对话历史的消息。
                用于注入少量示例或引导模型的响应风格。
                示例：[{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
            platform (str): 用户所在的接口平台（例如 "cli"、"telegram"、"discord"、"whatsapp"）。
                用于将平台特定的格式提示注入系统提示。
            skip_context_files (bool): 如果为 True，跳过 SOUL.md、AGENTS.md 和 .cursorrules 的自动注入
                到系统提示中。用于批处理和数据生成，以避免
                用用户特定的角色或项目指令污染轨迹。
        """
        _install_safe_stdio()

        self.model = model
        self.max_iterations = max_iterations
        # Shared iteration budget — parent creates, children inherit.
        # Consumed by every LLM turn across parent + all subagents.
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli", "telegram", "discord", "whatsapp", etc.
        # Pluggable print function — CLI replaces this with _cprint so that
        # raw ANSI status lines are routed through prompt_toolkit's renderer
        # instead of going directly to stdout where patch_stdout's StdoutProxy
        # would mangle the escape sequences.  None = use builtins.print.
        self._print_fn = None
        self.background_review_callback = None  # Optional sync callback for gateway delivery
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.persist_session = persist_session
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # Store effective base URL for feature detection (prompt caching, reasoning, etc.)
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and "chatgpt.com/backend-api/codex" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif self.provider == "anthropic" or (provider_name is None and "api.anthropic.com" in self._base_url_lower):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # Third-party Anthropic-compatible endpoints (e.g. MiniMax, DashScope)
            # use a URL convention ending in /anthropic. Auto-detect these so the
            # Anthropic Messages API adapter is used instead of chat completions.
            self.api_mode = "anthropic_messages"
        else:
            self.api_mode = "chat_completions"

        # Direct OpenAI sessions use the Responses API path.  GPT-5.x tool
        # calls with reasoning are rejected on /v1/chat/completions, and
        # Hermes is a tool-using client by default.
        if self.api_mode == "chat_completions" and self._is_direct_openai_url():
            self.api_mode = "codex_responses"

        # Pre-warm OpenRouter model metadata cache in a background thread.
        # fetch_model_metadata() is cached for 1 hour; this avoids a blocking
        # HTTP request on the first API response when pricing is estimated.
        if self.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self._reasoning_deltas_fired = False  # Set by _fire_reasoning_delta, reset per API call
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback
        self._last_reported_tool = None  # Track for "new tool" mode
        
        # Tool execution state — allows _vprint during tool execution
        # even when stream consumers are registered (no tokens streaming then)
        self._executing_tools = False

        # Interrupt mechanism for breaking out of tool loops
        self._interrupt_requested = False
        self._interrupt_message = None  # Optional message that triggered interrupt
        self._client_lock = threading.RLock()
        
        # Subagent delegation state
        self._delegate_depth = 0        # 0 = top-level agent, incremented for children
        self._active_children = []      # Running child AIAgents (for interrupt propagation)
        self._active_children_lock = threading.Lock()
        
        # 存储OpenRouter提供商偏好设置
        # 这些参数用于控制OpenRouter如何选择上游提供商
        self.providers_allowed = providers_allowed          # 允许的提供商列表
        self.providers_ignored = providers_ignored          # 忽略的提供商列表
        self.providers_order = providers_order              # 提供商尝试顺序
        self.provider_sort = provider_sort                  # 排序策略：price/throughput/latency
        self.provider_require_parameters = provider_require_parameters  # 要求参数支持
        self.provider_data_collection = provider_data_collection  # 数据收集策略

        # 存储工具集过滤选项
        # 工具集是工具的逻辑分组，可以按平台启用/禁用
        self.enabled_toolsets = enabled_toolsets    # 仅启用这些工具集中的工具
        self.disabled_toolsets = disabled_toolsets  # 禁用这些工具集中的工具

        # 模型响应配置
        self.max_tokens = max_tokens  # None = 使用模型默认值
        self.reasoning_config = reasoning_config  # None = 使用默认值（OpenRouter为medium）
        self.prefill_messages = prefill_messages or []  # 预填充的对话轮次

        # Anthropic提示缓存：通过OpenRouter为Claude模型自动启用。
        # 通过缓存对话前缀，在多轮对话中减少约75%的输入成本。
        # 使用system_and_3策略（4个断点）。
        is_openrouter = self._is_openrouter_url()  # 是否使用OpenRouter
        is_claude = "claude" in self.model.lower()  # 是否为Claude模型
        is_native_anthropic = self.api_mode == "anthropic_messages"  # 是否为原生Anthropic API
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic  # 启用提示缓存
        self._cache_ttl = "5m"  # 默认5分钟TTL（1.25倍写入成本）

        # 迭代预算压力：当LLM接近max_iterations时警告。
        # 警告被注入到最后的工具结果JSON中（而不是作为单独的消息），
        # 这样它们不会破坏消息结构或使缓存失效。
        self._budget_caution_threshold = 0.7   # 70% —— 提示开始收尾
        self._budget_warning_threshold = 0.9   # 90% —— 紧急，立即响应
        self._budget_pressure_enabled = True

        # 上下文压力警告：当上下文填满时通知用户（而不是LLM）。
        # 纯粹是信息性的 —— 在CLI输出中显示，并通过status_callback
        # 发送到网关平台。不会注入到消息中。
        self._context_pressure_warned = False

        # 活动跟踪系统 - 智能体行为监控和诊断支持
        # 活动跟踪 —— 在每次 API 调用、工具执行和流块时更新。
        # 由网关超时处理程序用于报告智能体被终止时在做什么，
        # 以及由"仍在工作"通知用于显示进度。
        self._last_activity_ts: float = time.time()  # 最后活动时间戳
        self._last_activity_desc: str = "initializing"  # 最后活动描述
        self._current_tool: str | None = None  # 当前正在执行的工具
        self._api_call_count: int = 0  # API 调用计数器

        # 集中式日志系统 - 智能体运行状态记录
        # 集中式日志 —— agent.log (INFO+) 和 errors.log (WARNING+)
        # 都位于 ~/.hermes/logs/。幂等的，因此网关模式
        # （每条消息创建一个新的 AIAgent）不会重复处理器。
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            # 详细日志模式：启用第三方库的详细日志
            setup_verbose_logging()
            logger.info("Verbose logging enabled (third-party library logs suppressed)")
        else:
            if self.quiet_mode:
                # 在安静模式下（CLI 默认），抑制所有工具/基础设施日志
                # 在 *控制台* 上的噪音。TUI 有自己的丰富显示
                # 用于状态；记录器 INFO/WARNING 消息只会使其混乱。
                # 文件处理器（agent.log、errors.log）仍然捕获所有内容。
                for quiet_logger in [
                    'tools',               # 所有 tools.*（终端、浏览器、网络、文件等）
                    'run_agent',            # 智能体运行器内部
                    'trajectory_compressor',  # 轨迹压缩器
                    'cron',                 # 调度器（仅在守护程序模式下相关）
                    'hermes_cli',           # CLI 助手
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # 内部流回调管理 - 流式传输支持
        # 内部流回调（在流式 TTS 期间设置）。
        # 在此处初始化，以便 _vprint 可以在 run_conversation 之前引用它。
        self._stream_callback = None
        # 延迟段落分隔标志 —— 在工具迭代后设置，以便
        # 单个 "\n\n" 被添加到下一个真实文本增量的前面。
        self._stream_needs_break = False

        # 用户消息覆盖机制 - API 与持久化的差异处理
        # 可选的当前轮次用户消息覆盖，当面向 API 的
        # 用户消息有意不同于持久化的记录时使用
        # （例如，CLI 语音模式仅为实时调用添加临时前缀）。
        self._persist_user_message_idx = None  # 持久化用户消息索引
        self._persist_user_message_override = None  # 持久化用户消息覆盖

        # Anthropic 图像回退缓存 - 性能优化机制
        # 缓存每个图像负载/URL 的 Anthropic 图像到文本回退，
        # 这样单个工具循环不会重复在同一图像历史上运行辅助视觉。
        # 这种缓存避免了重复的视觉处理调用，显著提升了性能。
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # LLM 客户端初始化 - 提供商路由和认证管理
        # 通过集中式提供商路由器初始化 LLM 客户端。
        # 路由器处理所有已知提供商的身份验证解析、基础 URL、标头、
        # 和 Codex/Anthropic 包装。
        # raw_codex=True，因为主智能体需要直接的 responses.stream()
        # 访问以进行 Codex Responses API 流式传输。
        self._anthropic_client = None  # Anthropic 客户端（延迟初始化）
        self._is_anthropic_oauth = False  # 是否使用 Anthropic OAuth

        if self.api_mode == "anthropic_messages":
            # Anthropic Messages API 模式 - 原生 Anthropic 支持
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            
            # 认证策略：仅在提供商确实是 Anthropic 时才回退到 ANTHROPIC_TOKEN。
            # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）必须使用自己的 API 密钥。
            # 回退会将 Anthropic 凭据发送到第三方端点（修复 #1739、#minimax-401）。
            _is_native_anthropic = self.provider == "anthropic"
            effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
            
            # 存储有效的认证信息
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url
            
            # 检测是否使用 OAuth 令牌
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            self._is_anthropic_oauth = _is_oat(effective_key)
            
            # 构建 Anthropic 客户端
            self._anthropic_client = build_anthropic_client(effective_key, base_url)
            
            # Anthropic 模式下不需要 OpenAI 客户端
            self.client = None
            self._client_kwargs = {}
            
            # 初始化成功提示（非安静模式）
            if not self.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                if effective_key and len(effective_key) > 12:
                    # 安全地显示令牌的一部分
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        else:
            # 非Anthropic Messages模式：使用OpenAI兼容的客户端
            if api_key and base_url:
                # 从CLI/网关显式提供凭据 —— 直接构建。
                # 运行时提供商解析器已经为我们处理了认证。
                client_kwargs = {"api_key": api_key, "base_url": base_url}

                # 特殊处理：Copilot ACP模式需要命令和参数
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args

                effective_base = base_url

                # 为特定提供商添加默认标头
                if "openrouter" in effective_base.lower():
                    # OpenRouter标头：标识应用和分类
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.githubcopilot.com" in effective_base.lower():
                    # GitHub Copilot需要特殊的标头
                    from hermes_cli.models import copilot_default_headers
                    client_kwargs["default_headers"] = copilot_default_headers()
                elif "api.kimi.com" in effective_base.lower():
                    # Kimi API需要自定义User-Agent
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.3",
                    }
            else:
                # 没有显式凭据 —— 使用集中式提供商路由器
                # 提供商路由器会自动解析API密钥、基础URL和其他配置
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    # 使用路由器返回的客户端配置
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    # 保留路由器设置的任何默认标头
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # 当用户明确选择了非OpenRouter提供商
                    # 但未找到凭据时，快速失败并显示清晰的消息，
                    # 而不是默默地通过OpenRouter路由。
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_explicit.upper()}_API_KEY environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    # 最终回退：尝试原始的OpenRouter密钥
                    client_kwargs = {
                        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                        "base_url": OPENROUTER_BASE_URL,
                        "default_headers": {
                            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                            "X-OpenRouter-Title": "Hermes Agent",
                            "X-OpenRouter-Categories": "productivity,cli-agent",
                        },
                    }

            self._client_kwargs = client_kwargs  # 存储以便中断后重建客户端

            # 为OpenRouter上的Claude启用细粒度工具流式传输。
            # 如果没有这个，Anthropic会缓冲整个工具调用，在思考时
            # 沉默数分钟 —— OpenRouter的上游代理在沉默期间超时。
            # beta标头使Anthropic逐token流式传输工具调用参数，
            # 保持连接活跃。
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if "openrouter" in _effective_base and "claude" in (self.model or "").lower():
                # 检查并添加细粒度工具流式传输的beta标头
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    # 如果已有其他beta特性，用逗号连接；否则直接设置
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            try:
                # 创建OpenAI客户端实例
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    # 显示初始化信息
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # 始终显示API密钥信息（已遮罩）以便调试认证问题
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # 提供商回退链配置 - 容错和可靠性保障
        # 提供商回退链 —— 当主要提供商耗尽时（速率限制、过载、连接失败）
        # 尝试的备份提供商的有序列表。
        # 支持传统的单个字典 fallback_model 和新的列表 fallback_providers 格式。
        if isinstance(fallback_model, list):
            # 新格式：提供商回退链列表
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            # 传统格式：单个回退模型配置
            self._fallback_chain = [fallback_model]
        else:
            # 没有配置回退
            self._fallback_chain = []
        
        # 回退状态管理
        self._fallback_index = 0  # 当前回退索引
        self._fallback_activated = False  # 回退是否已激活
        
        # 传统属性保留向后兼容（测试、外部调用者）
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        
        # 回退配置显示（非安静模式）
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 Fallback chain ({len(self._fallback_chain)} providers): " +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # 获取可用工具并进行过滤
        # 根据启用的工具集和禁用的工具集筛选工具
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )

        # 显示工具配置并存储有效的工具名称用于验证
        self.valid_tool_names = set()
        if self.tools:
            # 提取所有工具名称
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")

                # 如果应用了过滤，显示过滤信息
                if enabled_toolsets:
                    print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")

        # 检查工具要求（如API密钥、依赖项等）
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")

        # 显示轨迹保存状态
        if self.save_trajectories and not self.quiet_mode:
            print("📝 Trajectory saving enabled")

        # 显示临时系统提示状态
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")

        # 显示提示缓存状态
        if self._use_prompt_caching and not self.quiet_mode:
            source = "native Anthropic" if is_native_anthropic else "Claude via OpenRouter"
            print(f"💾 Prompt caching: ENABLED ({source}, {self._cache_ttl} TTL)")

        # 会话日志设置 - 自动保存对话轨迹用于调试
        self.session_start = datetime.now()
        if session_id:
            # 使用提供的会话ID（例如来自CLI）
            self.session_id = session_id
        else:
            # 生成新的会话ID：时间戳_短UUID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"

        # 会话日志存储在~/.hermes/sessions/，与网关会话并列
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"

        # 跟踪会话日志的对话消息
        self._session_messages: List[Dict[str, Any]] = []

        # 缓存的系统提示 —— 每个会话构建一次，仅在压缩时重建
        # 这是性能优化的关键：避免每次API调用都重新构建系统提示
        self._cached_system_prompt: Optional[str] = None

        # 文件系统检查点管理器（透明的 —— 不是工具）
        # 检查点用于保存和恢复文件系统状态，支持撤销操作
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )

        # SQLite会话存储（可选 —— 由CLI或网关提供）
        # 提供持久化的会话管理，支持会话搜索和历史
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # 跟踪DB写入游标以防止重复写入
        if self._session_db:
            try:
                # 在数据库中创建会话记录
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                    parent_session_id=self._parent_session_id,
                )
            except Exception as e:
                # 暂时的SQLite锁争用（例如CLI和网关并发写入）
                # 绝不能永久禁用此agent的session_search。
                # 保持_session_db存活 —— 一旦锁清除，后续的消息刷新
                # 和session_search调用仍然可以工作。
                # 会话行可能在此运行期间从索引中丢失，但这是可恢复的
                # （刷新会upsert行）。
                logger.warning(
                    "Session DB create_session failed (session_search still available): %s", e
                )

        # 内存中的待办事项列表，用于任务规划（每个agent/会话一个）
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()

        # 加载配置一次，用于memory、skills和compression部分
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}

        # 持久化内存系统（MEMORY.md + USER.md）—— 从磁盘加载
        # 这是Hermes的自改进核心：agent可以跨会话记住用户和上下文
        self._memory_store = None  # 内存存储实例
        self._memory_enabled = False  # 是否启用MEMORY.md
        self._user_profile_enabled = False  # 是否启用USER.md
        self._memory_nudge_interval = 10  # 内存提醒间隔（轮次）
        self._memory_flush_min_turns = 6  # 内存刷新的最小轮次
        self._turns_since_memory = 0  # 自上次内存操作以来的轮次
        self._iters_since_skill = 0  # 自上次技能操作以来的迭代次数
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),  # MEMORY.md字符限制
                        user_char_limit=mem_config.get("user_char_limit", 1375),  # USER.md字符限制
                    )
                    self._memory_store.load_from_disk()  # 从磁盘加载现有内存
            except Exception:
                pass  # 内存是可选的 —— 不要破坏agent初始化



        # 内存提供商插件（外部的 —— 一次一个，与内置的并列）
        # 从config读取memory.provider以选择要激活的插件。
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                # 自动迁移：如果Honcho被主动配置（启用+凭据）
                # 但memory.provider未设置，自动激活honcho插件。
                # 仅拥有配置文件是不够的 —— 用户可能已禁用Honcho或
                # 文件可能来自不同的工具。
                if not _mem_provider_name:
                    try:
                        from plugins.memory.honcho.client import HonchoClientConfig as _HCC
                        _hcfg = _HCC.from_global_config()
                        if _hcfg.enabled and (_hcfg.api_key or _hcfg.base_url):
                            _mem_provider_name = "honcho"
                            # 持久化，以便这只自动迁移一次
                            try:
                                from hermes_cli.config import load_config as _lc, save_config as _sc
                                _cfg = _lc()
                                _cfg.setdefault("memory", {})["provider"] = "honcho"
                                _sc(_cfg)
                            except Exception:
                                pass
                            if not self.quiet_mode:
                                print("  ✓ Auto-migrated Honcho to memory provider plugin.")
                                print("    Your config and data are preserved.\n")
                    except Exception:
                        pass

                if _mem_provider_name:
                    # 加载并初始化内存提供商插件
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        # 初始化内存管理器，传递会话上下文
                        from hermes_constants import get_hermes_home as _ghh
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(_ghh()),
                            "agent_context": "primary",
                        }
                        # Profile身份用于按profile范围划分提供商
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("Memory provider '%s' activated", _mem_provider_name)
                    else:
                        logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("Memory provider plugin init failed: %s", _mpe)
                self._memory_manager = None

        # 将内存提供商工具schema注入工具表面
        # 这允许内存提供商提供自己的工具（如Honcho的方言用户建模）
        if self._memory_manager and self.tools is not None:
            for _schema in self._memory_manager.get_all_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)

        # 技能配置：技能创建提醒的提醒间隔
        # 技能是agent从经验中学习的核心机制
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
        except Exception:
            pass

        # 工具使用强制配置："auto"（默认 —— 匹配硬编码模型列表），
        # true（始终），false（从不），或子字符串列表。
        # 用于强制某些模型使用工具而不是纯文本响应
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # 初始化上下文压缩器，用于自动上下文管理
        # 当接近模型的上下文限制时压缩对话
        # 通过config.yaml（compression部分）配置
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))  # 压缩阈值：50%
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
        compression_summary_model = _compression_cfg.get("summary_model") or None  # 用于摘要的模型
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))  # 目标压缩比：20%
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))  # 保护最后N条消息

        # 从模型配置读取显式的context_length覆盖
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                _config_context_length = None

        # 检查custom_providers中每个模型的context_length
        # 这允许用户为自定义端点指定上下文长度
        if _config_context_length is None:
            _custom_providers = _agent_cfg.get("custom_providers")
            if isinstance(_custom_providers, list):
                for _cp_entry in _custom_providers:
                    if not isinstance(_cp_entry, dict):
                        continue
                    _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                    if _cp_url and _cp_url == self.base_url.rstrip("/"):
                        _cp_models = _cp_entry.get("models", {})
                        if isinstance(_cp_models, dict):
                            _cp_model_cfg = _cp_models.get(self.model, {})
                            if isinstance(_cp_model_cfg, dict):
                                _cp_ctx = _cp_model_cfg.get("context_length")
                                if _cp_ctx is not None:
                                    try:
                                        _config_context_length = int(_cp_ctx)
                                    except (TypeError, ValueError):
                                        pass
                        break

        # 创建上下文压缩器实例
        self.context_compressor = ContextCompressor(
            model=self.model,
            threshold_percent=compression_threshold,
            protect_first_n=3,  # 始终保护前3条消息（系统提示等）
            protect_last_n=compression_protect_last,
            summary_target_ratio=compression_target_ratio,
            summary_model_override=compression_summary_model,
            quiet_mode=self.quiet_mode,
            base_url=self.base_url,
            api_key=getattr(self, "api_key", ""),
            config_context_length=_config_context_length,
            provider=self.provider,
        )
        self.compression_enabled = compression_enabled
        # 子目录提示追踪器：跟踪工作目录变化
        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0  # 用户轮次计数

        # 会话的累积token使用统计
        # 这些统计用于成本估算和资源监控
        self.session_prompt_tokens = 0  # 提示token总数
        self.session_completion_tokens = 0  # 完成token总数
        self.session_total_tokens = 0  # 总token数
        self.session_api_calls = 0  # API调用次数
        self.session_input_tokens = 0  # 输入token数
        self.session_output_tokens = 0  # 输出token数
        self.session_cache_read_tokens = 0  # 缓存读取token数（提示缓存）
        self.session_cache_write_tokens = 0  # 缓存写入token数
        self.session_reasoning_tokens = 0  # 推理token数（Claude的thinking）
        self.session_estimated_cost_usd = 0.0  # 估算成本（美元）
        self.session_cost_status = "unknown"  # 成本状态：unknown/estimated/actual
        self.session_cost_source = "none"  # 成本来源：none/models.dev/usage

        # 显示上下文限制信息
        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (compress at {int(compression_threshold*100)}% = {self.context_compressor.threshold_tokens:,})")
            else:
                print(f"📊 Context limit: {self.context_compressor.context_length:,} tokens (auto-compression disabled)")

        # Snapshot primary runtime for per-turn restoration.  When fallback
        # activates during a turn, the next turn restores these values so the
        # preferred model gets a fresh attempt each time.  Uses a single dict
        # so new state fields are easy to add without N individual attributes.
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            # Compressor state that _try_activate_fallback() overwrites
            "compressor_model": _cc.model,
            "compressor_base_url": _cc.base_url,
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": _cc.provider,
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

    def reset_session_state(self):
        """Reset all session-scoped token counters to 0 for a fresh session.
        
        This method encapsulates the reset logic for all session-level metrics
        including:
        - Token usage counters (input, output, total, prompt, completion)
        - Cache read/write tokens
        - API call count
        - Reasoning tokens
        - Estimated cost tracking
        - Context compressor internal counters
        
        The method safely handles optional attributes (e.g., context compressor)
        using ``hasattr`` checks.
        
        This keeps the counter reset logic DRY and maintainable in one place
        rather than scattering it across multiple methods.
        """
        # Token usage counters
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Context compressor internal counters (if present)
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.last_prompt_tokens = 0
            self.context_compressor.last_completion_tokens = 0
            self.context_compressor.last_total_tokens = 0
            self.context_compressor.compression_count = 0
            self.context_compressor._context_probed = False
            self.context_compressor._context_probe_persistable = False
            # Iterative summary from previous session must not bleed into new one (#2635)
            self.context_compressor._previous_summary = None
    
    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """Switch the model/provider in-place for a live agent.

        Called by the /model command handlers (CLI and gateway) after
        ``model_switch.switch_model()`` has resolved credentials and
        validated the model.  This method performs the actual runtime
        swap: rebuilding clients, updating caching flags, and refreshing
        the context compressor.

        The implementation mirrors ``_try_activate_fallback()`` for the
        client-swap logic but also updates ``_primary_runtime`` so the
        change persists across turns (unlike fallback which is
        turn-scoped).
        """
        import logging
        from hermes_cli.providers import determine_api_mode

        # ── Determine api_mode if not provided ──
        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        old_model = self.model
        old_provider = self.provider

        # ── Swap core runtime fields ──
        self.model = new_model
        self.provider = new_provider
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode
        if api_key:
            self.api_key = api_key

        # ── Build new client ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            effective_key = api_key or self.api_key or resolve_anthropic_token() or ""
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key)
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        # ── Re-evaluate prompt caching ──
        is_native_anthropic = api_mode == "anthropic_messages"
        self._use_prompt_caching = (
            ("openrouter" in (self.base_url or "").lower() and "claude" in new_model.lower())
            or is_native_anthropic
        )

        # ── Update context compressor ──
        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
            )
            self.context_compressor.model = self.model
            self.context_compressor.base_url = self.base_url
            self.context_compressor.api_key = self.api_key
            self.context_compressor.provider = self.provider
            self.context_compressor.context_length = new_context_length
            self.context_compressor.threshold_tokens = int(
                new_context_length * self.context_compressor.threshold_percent
            )

        # ── Invalidate cached system prompt so it rebuilds next turn ──
        self._cached_system_prompt = None

        # ── Update _primary_runtime so the change persists across turns ──
        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "compressor_model": _cc.model if _cc else self.model,
            "compressor_base_url": _cc.base_url if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": _cc.provider if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        # ── Reset fallback state ──
        self._fallback_activated = False
        self._fallback_index = 0

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )

    def _safe_print(self, *args, **kwargs):
        """Print that silently handles broken pipes / closed stdout.

        In headless environments (systemd, Docker, nohup) stdout may become
        unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
        can crash cron jobs and lose completed work.

        Internally routes through ``self._print_fn`` (default: builtin
        ``print``) so callers such as the CLI can inject a renderer that
        handles ANSI escape sequences properly (e.g. prompt_toolkit's
        ``print_formatted_text(ANSI(...))``) without touching this method.
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed when actively streaming tokens.

        Pass ``force=True`` for error/warning messages that should always be
        shown even during streaming playback (TTS or display).

        During tool execution (``_executing_tools`` is True), printing is
        allowed even with stream consumers registered because no tokens
        are being streamed at that point.

        After the main response has been delivered and the remaining tool
        calls are post-response housekeeping (``_mute_post_response``),
        all non-forced output is suppressed.
        """
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """Return True when quiet-mode spinner output has a safe sink.

        In headless/stdio-protocol environments, a raw spinner with no custom
        ``_print_fn`` falls back to ``sys.stdout`` and can corrupt protocol
        streams such as ACP JSON-RPC. Allow quiet spinners only when either:
        - output is explicitly rerouted via ``_print_fn``; or
        - stdout is a real TTY.
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _emit_status(self, message: str) -> None:
        """Emit a lifecycle status message to both CLI and gateway channels.

        CLI users see the message via ``_vprint(force=True)`` so it is always
        visible regardless of verbose/quiet mode.  Gateway consumers receive
        it through ``status_callback("lifecycle", ...)``.

        This helper never raises — exceptions are swallowed so it cannot
        interrupt the retry/fallback logic.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return "openrouter" in self._base_url_lower

    def _is_anthropic_url(self) -> bool:
        """Return True when the base URL targets Anthropic (native or /anthropic proxy path)."""
        return "api.anthropic.com" in self._base_url_lower or self._base_url_lower.rstrip("/").endswith("/anthropic")

    def _max_tokens_param(self, value: int) -> dict:
        """Return the correct max tokens kwarg for the current provider.
        
        OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
        'max_completion_tokens'. OpenRouter, local models, and older
        OpenAI models use 'max_tokens'.
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        Check if content has actual text after any reasoning/thinking blocks.

        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        Must stay in sync with _strip_think_blocks() tag variants.

        Args:
            content: The assistant message content to check

        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False

        # Remove all reasoning tag variants (must match _strip_think_blocks)
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())
    
    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text."""
        if not content:
            return ""
        # Strip all reasoning tag variants: <think>, <thinking>, <THINKING>,
        # <reasoning>, <REASONING_SCRATCHPAD>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL)
        content = re.sub(r'</?(?:think|thinking|reasoning|REASONING_SCRATCHPAD)>\s*', '', content, flags=re.IGNORECASE)
        return content

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """Detect a planning/ack message that should continue instead of ending the turn."""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action
    
    
    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        Extract reasoning/thinking content from an assistant message.
        
        OpenRouter and various providers can return reasoning in multiple formats:
        1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
        2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
        3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)
        
        Args:
            assistant_message: The assistant message object from the API response
            
        Returns:
            Combined reasoning text, or None if no reasoning found
        """
        reasoning_parts = []
        
        # Check direct reasoning field
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # Check reasoning_content field (alternative name used by some providers)
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # Don't duplicate if same as reasoning
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # Check reasoning_details array (OpenRouter unified format)
        # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # Extract summary from reasoning detail object
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # Some providers embed reasoning directly inside assistant content
        # instead of returning structured reasoning fields.  Only fall back
        # to inline extraction when no structured reasoning was found.
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # Combine all reasoning parts
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None

    def _classify_empty_content_response(
        self,
        assistant_message,
        *,
        finish_reason: Optional[str],
        approx_tokens: int,
        api_messages: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Classify think-only/empty responses so we can retry, compress, or salvage.

        We intentionally do NOT short-circuit all structured-reasoning responses.
        Prior discussion/PR history shows some models recover on retry. Instead we:
        - compress immediately when the pattern looks like implicit context pressure
        - salvage reasoning early when the same reasoning-only payload repeats
        - otherwise preserve the normal retry path
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        has_structured_reasoning = bool(
            getattr(assistant_message, "reasoning", None)
            or getattr(assistant_message, "reasoning_content", None)
            or getattr(assistant_message, "reasoning_details", None)
        )
        content = getattr(assistant_message, "content", None) or ""
        stripped_content = self._strip_think_blocks(content).strip()
        signature = (
            content,
            reasoning_text or "",
            bool(has_structured_reasoning),
            finish_reason or "",
        )
        repeated_signature = signature == getattr(self, "_last_empty_content_signature", None)

        compressor = getattr(self, "context_compressor", None)
        ctx_len = getattr(compressor, "context_length", 0) or 0
        threshold_tokens = getattr(compressor, "threshold_tokens", 0) or 0
        is_large_session = bool(
            (ctx_len and approx_tokens >= max(int(ctx_len * 0.4), threshold_tokens))
            or len(api_messages) > 80
        )
        is_local_custom = is_local_endpoint(getattr(self, "base_url", "") or "")
        is_resumed = bool(conversation_history)
        context_pressure_signals = any(
            [
                finish_reason == "length",
                getattr(compressor, "_context_probed", False),
                is_large_session,
                is_resumed,
            ]
        )
        should_compress = bool(
            self.compression_enabled
            and is_local_custom
            and context_pressure_signals
            and not stripped_content
        )

        self._last_empty_content_signature = signature
        return {
            "reasoning_text": reasoning_text,
            "has_structured_reasoning": has_structured_reasoning,
            "repeated_signature": repeated_signature,
            "should_compress": should_compress,
            "is_local_custom": is_local_custom,
            "is_large_session": is_large_session,
            "is_resumed": is_resumed,
        }
    
    def _cleanup_task_resources(self, task_id: str) -> None:
        """Clean up VM and browser resources for a given task."""
        try:
            cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # Background memory/skill review
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "Review the conversation above and consider saving to memory if appropriate.\n\n"
        "Focus on:\n"
        "1. Has the user revealed things about themselves — their persona, desires, "
        "preferences, or personal details worth remembering?\n"
        "2. Has the user expressed expectations about how you should behave, their work "
        "style, or ways they want you to operate?\n\n"
        "If something stands out, save it using the memory tool. "
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _SKILL_REVIEW_PROMPT = (
        "Review the conversation above and consider saving or updating a skill if appropriate.\n\n"
        "Focus on: was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome?\n\n"
        "If a relevant skill already exists, update it with what you learned. "
        "Otherwise, create a new skill if the approach is reusable.\n"
        "If nothing is worth saving, just say 'Nothing to save.' and stop."
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves — their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "If so, save using the memory tool.\n\n"
        "**Skills**: Was a non-trivial approach used to complete a task that required trial "
        "and error, or changing course due to experiential findings along the way, or did "
        "the user expect or desire a different method or outcome? If a relevant skill "
        "already exists, update it. Otherwise, create a new one if the approach is reusable.\n\n"
        "Only act if there's something genuinely worth saving. "
        "If nothing stands out, just say 'Nothing to save.' and stop."
    )

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """Spawn a background thread to review the conversation for memory/skill saves.

        Creates a full AIAgent fork with the same model, tools, and context as the
        main session. The review prompt is appended as the next user turn in the
        forked conversation. Writes directly to the shared memory/skill stores.
        Never modifies the main conversation history or produces user-visible output.
        """
        import threading

        # Pick the right prompt based on which triggers fired
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib, os as _os
            review_agent = None
            try:
                with open(_os.devnull, "w") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # Scan the review agent's messages for successful tool actions
                # and surface a compact summary to the user.
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (target and "add" in message.lower()):
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "Entry added" in message:
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = "Memory" if target == "memory" else "User profile" if target == "user" else target
                        actions.append(f"{label} updated")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(f"💾 {summary}")
                        except Exception:
                            pass

            except Exception as e:
                logger.debug("Background memory/skill review failed: %s", e)
            finally:
                # Explicitly close the OpenAI/httpx client so GC doesn't
                # try to clean it up on a dead asyncio event loop (which
                # produces "Event loop is closed" errors in the terminal).
                if review_agent is not None:
                    client = getattr(review_agent, "client", None)
                    if client is not None:
                        try:
                            review_agent._close_openai_client(
                                client, reason="bg_review_done", shared=True
                            )
                            review_agent.client = None
                        except Exception:
                            pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message before persistence/return.

        Some call paths need an API-only user-message variant without letting
        that synthetic text leak into persisted transcripts or resumed session
        history. When an override is configured for the active turn, mutate the
        in-memory messages list in place so both persistence and returned
        history stay clean.
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save session state to both JSON log and SQLite on any exit path.

        Ensures conversations are never lost, even on errors or early returns.
        Skipped when ``persist_session=False`` (ephemeral helper flows).
        """
        if not self.persist_session:
            return
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Persist any un-flushed messages to the SQLite session store.

        Uses _last_flushed_db_idx to track which messages have already been
        written, so repeated calls (from multiple exit paths) only write
        truly new messages — preventing the duplicate-write bug (#860).
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # If create_session() failed at startup (e.g. transient lock), the
            # session row may not exist yet.  ensure_session() uses INSERT OR
            # IGNORE so it is a no-op when the row is already there.
            self._session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.
        
        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.
        
        Args:
            messages: Full message list
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()
        
        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]
    
    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.
        
        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"
        
        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    
    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        Convert internal message format to trajectory format for saving.
        
        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
            
        Returns:
            List[Dict]: Messages in trajectory format
        """
        trajectory = []
        
        # Add system message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # Add the actual user prompt (from the dataset) as the first human message
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # Skip the first message (the user query) since we already added it above.
        # Prefill messages are injected at API-call time only (not in the messages
        # list), so no offset adjustment is needed here.
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                        # (used when native thinking is disabled and model reasons via XML)
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    # so the format is consistent for training data
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # Collect all subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = "<tool_response>\n"
                        
                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # Skip the tool messages we just processed
                
                else:
                    # Regular assistant message without tool calls
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available (native thinking tokens)
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    # Ensure every gpt turn has a <think> block (empty if no reasoning)
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    
    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        Save conversation trajectory to JSONL file.
        
        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)
    
    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """Extract a human-readable one-liner from an API error.

        Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
        <title> tag instead of dumping raw HTML.  Falls back to a truncated
        str(error) for everything else.
        """
        import re as _re
        raw = str(error)

        # Cloudflare / proxy HTML pages: grab the <title> for a clean summary
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = _re.search(r"<title[^>]*>([^<]+)</title>", raw, _re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # Also grab Cloudflare Ray ID if present
            ray = _re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # JSON body errors from OpenAI/Anthropic SDKs
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # Fallback: truncate the raw string but give more room than 200 chars
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """
        Clean up error messages for user display, removing HTML content and truncating.
        
        Args:
            error_msg: Raw error message from API or exception
            
        Returns:
            Clean, user-friendly error message
        """
        if not error_msg:
            return "Unknown error"
            
        # Remove HTML content (common with CloudFlare and gateway error pages)
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "Service temporarily unavailable (HTML error page returned)"
            
        # Remove newlines and excessive whitespace
        cleaned = ' '.join(error_msg.split())
        
        # Truncate if too long
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
            
        return cleaned

    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """Extract structured rate-limit details from provider errors."""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        Dump a debug-friendly HTTP request record for the active inference API.

        Captures the request body from api_kwargs (excluding transport-only keys
        like timeout). Intended for debugging provider-side 4xx failures where
        retries are not useful.
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        Save the full raw session to a JSON file.

        Stores every message exactly as the agent sees it: user messages,
        assistant messages (with reasoning, finish_reason, tool_calls),
        tool responses (with tool_call_id, tool_name), and injected system
        messages (compression summaries, todo snapshots, etc.).

        REASONING_SCRATCHPAD tags are converted to <think> blocks for consistency.
        Overwritten after each turn so it always reflects the latest state.
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # Clean assistant content for session logs
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # Guard: never overwrite a larger session log with fewer messages.
            # This protects against data loss when --resume loads a session whose
            # messages weren't fully written to SQLite — the resumed agent starts
            # with partial history and would otherwise clobber the full JSON log.
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "Skipping session log overwrite: existing has %d messages, current has %d",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # corrupted existing file — allow the overwrite

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    
    def interrupt(self, message: str = None) -> None:
        """
        Request the agent to interrupt its current tool-calling loop.
        
        Call this from another thread (e.g., input handler, message receiver)
        to gracefully stop the agent and process a new message.
        
        Also signals long-running tool executions (e.g. terminal commands)
        to terminate early, so the agent can respond immediately.
        
        Args:
            message: Optional new message that triggered the interrupt.
                     If provided, the agent will include this in its response context.
        
        Example (CLI):
            # In a separate input thread:
            if user_typed_something:
                agent.interrupt(user_input)
        
        Example (Messaging):
            # When new message arrives for active session:
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # Signal all tools to abort any in-flight operations immediately
        _set_interrupt(True)
        # Propagate interrupt to any running child agents (subagent delegation)
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """Clear any pending interrupt request and the global tool interrupt signal."""
        self._interrupt_requested = False
        self._interrupt_message = None
        _set_interrupt(False)

    def _touch_activity(self, desc: str) -> None:
        """Update the last-activity timestamp and description (thread-safe)."""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc

    def get_activity_summary(self) -> dict:
        """Return a snapshot of the agent's current activity for diagnostics.

        Called by the gateway timeout handler to report what the agent was doing
        when it was killed, and by the periodic "still working" notifications.
        """
        elapsed = time.time() - self._last_activity_ts
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
        }

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down the memory provider — call at actual session boundaries.

        This calls on_session_end() then shutdown_all() on the memory
        manager. NOT called per-turn — only at CLI exit, /reset, gateway
        session expiry, etc.
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
    
    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        Recover todo state from conversation history.
        
        The gateway creates a fresh AIAgent per message, so the in-memory
        TodoStore is empty. We scan the history for the most recent todo
        tool response and replay it to reconstruct the state.
        """
        # Walk history backwards to find the most recent todo tool response
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Quick check: todo responses contain "todos" key
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # Replay the items into the store (replace mode)
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 Restored {len(last_todo_response)} todo item(s) from history")
        _set_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._interrupt_requested










    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        Assemble the full system prompt from all layers.
        
        Called once per session (cached on self._cached_system_prompt) and only
        rebuilt after context compression events. This ensures the system prompt
        is stable across all turns in a session, maximizing prefix cache hits.
        """
        # Layers (in order):
        #   1. Agent identity — SOUL.md when available, else DEFAULT_AGENT_IDENTITY
        #   2. User / gateway system prompt (if provided)
        #   3. Persistent memory (frozen snapshot)
        #   4. Skills guidance (if skills tools are loaded)
        #   5. Context files (AGENTS.md, .cursorrules — SOUL.md excluded here when used as identity)
        #   6. Current date & time (frozen at build time)
        #   7. Platform-specific formatting hint

        # Try SOUL.md as primary identity (unless context files are skipped)
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # Fallback to hardcoded identity
            _ai_peer_name = (
                None
                if False
                else None
            )
            if _ai_peer_name:
                _identity = DEFAULT_AGENT_IDENTITY.replace(
                    "You are Hermes Agent",
                    f"You are {_ai_peer_name}",
                    1,
                )
            else:
                _identity = DEFAULT_AGENT_IDENTITY
            prompt_parts = [_identity]

        # Tool-aware behavioral guidance: only inject when the tools are loaded
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
        # Tool-use enforcement: tells the model to actually call tools instead
        # of describing intended actions.  Controlled by config.yaml
        # agent.tool_use_enforcement:
        #   "auto" (default) — matches TOOL_USE_ENFORCEMENT_MODELS
        #   true  — always inject (all models)
        #   false — never inject
        #   list  — custom model-name substrings to match
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # so it can refer the user to them rather than reinventing answers.

        # Note: ephemeral_system_prompt is NOT included here. It's injected at
        # API-call time only so it stays out of the cached/stored system prompt.
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md is always included when enabled.
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # External memory provider system prompt block (additive to built-in)
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            # Use TERMINAL_CWD for context file discovery when set (gateway
            # mode).  The gateway process runs from the hermes-agent install
            # dir, so os.getcwd() would pick up the repo's AGENTS.md and
            # other dev files — inflating token usage by ~10k for no benefit.
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\nSession ID: {self.session_id}"
        if self.model:
            timestamp_line += f"\nModel: {self.model}"
        if self.provider:
            timestamp_line += f"\nProvider: {self.provider}"
        prompt_parts.append(timestamp_line)

        # Alibaba Coding Plan API always returns "glm-4.7" as model name regardless
        # of the requested model. Inject explicit model identity into the system prompt
        # so the agent can correctly report which model it is (workaround for API bug).
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"You are powered by the model named {_model_short}. "
                f"The exact model ID is {self.model}. "
                f"When asked what model you are, always answer based on this information, "
                f"not on any model name returned by the API."
            )

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(prompt_parts)

    # =========================================================================
    # Pre/post-call guardrails (inspired by PR #1321 — @alireza78a)
    # =========================================================================

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """Extract call ID from a tool_call entry (dict or object)."""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs before every LLM call.

        Runs unconditionally — not gated on whether the context compressor
        is present — so orphans from session loading or manual message
        manipulation are always caught.
        """
        # --- Role allowlist: drop messages with roles the API won't accept ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in AIAgent._VALID_API_ROLES:
                logger.debug(
                    "Pre-call sanitizer: dropping message with invalid role %r",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Drop tool results with no matching assistant call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "Pre-call sanitizer: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. Inject stub results for calls whose result was dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result unavailable — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "Pre-call sanitizer: added %d stub tool result(s)",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """Truncate excess delegate_task calls to MAX_CONCURRENT_CHILDREN.

        The delegate_tool caps the task list inside a single call, but the
        model can emit multiple separate delegate_task tool_calls in one
        turn.  This truncates the excess, preserving all non-delegate calls.

        Returns the original list if no truncation was needed.
        """
        from tools.delegate_tool import MAX_CONCURRENT_CHILDREN
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= MAX_CONCURRENT_CHILDREN:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < MAX_CONCURRENT_CHILDREN:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "Truncated %d excess delegate_task call(s) to enforce "
            "MAX_CONCURRENT_CHILDREN=%d limit",
            delegate_count - MAX_CONCURRENT_CHILDREN, MAX_CONCURRENT_CHILDREN,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """Remove duplicate (tool_name, arguments) pairs within a single turn.

        Only the first occurrence of each unique pair is kept.
        Returns the original list if no duplicates were found.
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("Removed duplicate tool call: %s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """Attempt to repair a mismatched tool name before aborting.

        1. Try lowercase
        2. Try normalized (lowercase + hyphens/spaces -> underscores)
        3. Try fuzzy match (difflib, cutoff=0.7)

        Returns the repaired name if found in valid_tool_names, else None.
        """
        from difflib import get_close_matches

        # 1. Lowercase
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. Normalize
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. Fuzzy match
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """
        Invalidate the cached system prompt, forcing a rebuild on the next turn.
        
        Called after context compression events. Also reloads memory from disk
        so the rebuilt prompt captures any writes from this session.
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """Convert chat-completions tool schemas to Responses function-tool schemas."""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """Generate a deterministic call_id from tool call content.

        Used as a fallback when the API doesn't provide a call_id.
        Deterministic IDs prevent cache invalidation — random UUIDs would
        make every API call's prefix unique, breaking OpenAI's prompt cache.
        """
        import hashlib
        seed = f"{fn_name}:{arguments}:{index}"
        digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"call_{digest}"

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """Split a stored tool id into (call_id, response_item_id)."""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """Build a valid Responses `function_call.id` (must start with `fc_`)."""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal chat-style messages to Responses input items."""
        items: List[Dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # Replay encrypted reasoning items from previous turns
                    # so the API can maintain coherent reasoning chains.
                    codex_reasoning = msg.get("codex_reasoning_items")
                    has_codex_reasoning = False
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                items.append(ri)
                                has_codex_reasoning = True

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})
                    elif has_codex_reasoning:
                        # The Responses API requires a following item after each
                        # reasoning item (otherwise: missing_following_item error).
                        # When the assistant produced only reasoning with no visible
                        # content, emit an empty assistant message as the required
                        # following item.
                        items.append({"role": "assistant", "content": ""})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(
                                tc.get("id")
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                                else:
                                    _raw_args = str(fn.get("arguments", "{}"))
                                    call_id = self._deterministic_call_id(fn_name, _raw_args, len(items))
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": arguments,
                            })
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })

        return items

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        reasoning_item["id"] = item_id
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex Responses request 'model' must be a non-empty string.")
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Codex Responses request 'tools' must be a list when provided.")
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model", "instructions", "input", "tools", "store",
            "reasoning", "include", "max_output_tokens", "temperature",
            "tool_choice", "parallel_tool_calls", "prompt_cache_key",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "store": False,
        }
        if normalized_tools is not None:
            normalized["tools"] = normalized_tools

        # Pass through reasoning config
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include

        # Pass through max_output_tokens and temperature
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # Pass through tool_choice, parallel_tool_calls, prompt_cache_key
        for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

        unexpected = sorted(key for key in api_kwargs.keys() if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """Extract assistant text from a Responses message output item."""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """Extract a compact reasoning text from a Responses reasoning item."""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
        """Normalize a Responses API object to an assistant_message-like object."""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            # The Codex backend can return empty output when the answer was
            # delivered entirely via stream events. Check output_text as a
            # last-resort fallback before raising.
            out_text = getattr(response, "output_text", None)
            if isinstance(out_text, str) and out_text.strip():
                logger.debug(
                    "Codex response has empty output but output_text is present (%d chars); "
                    "synthesizing output item.", len(out_text.strip()),
                )
                output = [SimpleNamespace(
                    type="message", role="assistant", status="completed",
                    content=[SimpleNamespace(type="output_text", text=out_text.strip())],
                )]
                response.output = output
            else:
                raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                # Capture the full reasoning item for multi-turn continuity.
                # encrypted_content is an opaque blob the API needs back on
                # subsequent turns to maintain coherent reasoning chains.
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # Capture summary — required by the API when replaying reasoning items
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            # Response contains only reasoning (encrypted thinking state) with
            # no visible content or tool calls.  The model is still thinking and
            # needs another turn to produce the actual answer.  Marking this as
            # "stop" would send it into the empty-content retry loop which burns
            # 3 retries then fails — treat it as incomplete instead so the Codex
            # continuation path handles it correctly.
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """Check if an OpenAI client is closed.

        Handles both property and method forms of is_closed:
        - httpx.Client.is_closed is a bool property
        - openai.OpenAI.is_closed is a method returning bool

        Prior bug: getattr(client, "is_closed", False) returned the bound method,
        which is always truthy, causing unnecessary client recreation on every call.
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # Handle method (openai SDK) vs property (httpx)
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

        When a provider drops a connection mid-stream, httpx's ``client.close()``
        performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
        OS times them out (often minutes).  This method walks the httpx transport
        pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
        force an immediate TCP RST, freeing the file descriptors.

        Returns the number of sockets force-closed.
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx uses httpcore connection pools; connections live in
            # _connections (list) or _pool (list) depending on version.
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # Force-close TCP sockets first to prevent CLOSE-WAIT accumulation,
        # then do the graceful SDK-level close.
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """Detect and clean up dead TCP connections on the primary client.

        Inspects the httpx connection pool for sockets in unhealthy states
        (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
        and rebuilds the primary client from scratch.

        Returns True if dead connections were found and cleaned up.
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # Check for connections that are idle but have closed sockets
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # Probe socket health with a non-blocking recv peek
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # No data available — socket is healthy
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """Execute one streaming Responses API request and return the final response."""
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        self._reasoning_deltas_fired = False
        # Accumulate streamed text so we can recover if get_final_response()
        # returns empty output (e.g. chatgpt.com backend-api sends
        # response.incomplete instead of response.completed).
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # Fire callbacks on text content deltas (suppress during tool calls)
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # Track tool calls to suppress text streaming
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # Fire reasoning callbacks
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                        # Collect completed output items — some backends
                        # (chatgpt.com/backend-api/codex) stream valid items
                        # via response.output_item.done but the SDK's
                        # get_final_response() returns an empty output list.
                        elif event_type == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None:
                                collected_output_items.append(done_item)
                        # Log non-completed terminal events for diagnostics
                        elif event_type in ("response.incomplete", "response.failed"):
                            resp_obj = getattr(event, "response", None)
                            status = getattr(resp_obj, "status", None) if resp_obj else None
                            incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                            logger.warning(
                                "Codex Responses stream received terminal event %s "
                                "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                                event_type, status, incomplete_details,
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # PATCH: ChatGPT Codex backend streams valid output items
                    # but get_final_response() can return an empty output list.
                    # Backfill from collected items or synthesize from deltas.
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Fallback path for stream completion edge cases on Codex-style Responses backends."""
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._preflight_codex_api_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # Compatibility shim for mocks or providers that still return a concrete response.
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        collected_output_items: list = []
        collected_text_deltas: list = []
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

                # Collect output items and text deltas for backfill
                if event_type == "response.output_item.done":
                    done_item = getattr(event, "item", None)
                    if done_item is None and isinstance(event, dict):
                        done_item = event.get("item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif event_type in ("response.output_text.delta",):
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(delta)

                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    # Backfill empty output from collected stream events
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            terminal_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex fallback stream: backfilled %d output items",
                                len(collected_output_items),
                            )
                        elif collected_text_deltas:
                            assembled = "".join(collected_text_deltas)
                            terminal_response.output = [SimpleNamespace(
                                type="message", role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex fallback stream: synthesized from %d deltas (%d chars)",
                                len(collected_text_deltas), len(assembled),
                            )
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(new_token, getattr(self, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth)
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _OR_HEADERS

        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.3"}
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(runtime_key, runtime_base)
            self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """Attempt credential recovery via pool rotation.

        Returns (recovered, has_retried_429).
        On 429: first occurrence retries same credential (sets flag True).
                second consecutive 429 rotates to next credential (resets flag).
        On 402: immediately rotates (billing exhaustion won't resolve with retry).
        On 401: attempts token refresh before rotating.
        """
        pool = self._credential_pool
        if pool is None or status_code is None:
            return False, has_retried_429

        if status_code == 402:
            next_entry = pool.mark_exhausted_and_rotate(status_code=402, error_context=error_context)
            if next_entry is not None:
                logger.info(f"Credential 402 (billing) — rotated to pool entry {getattr(next_entry, 'id', '?')}")
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if status_code == 429:
            if not has_retried_429:
                return False, True
            next_entry = pool.mark_exhausted_and_rotate(status_code=429, error_context=error_context)
            if next_entry is not None:
                logger.info(f"Credential 429 (rate limit) — rotated to pool entry {getattr(next_entry, 'id', '?')}")
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if status_code == 401:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential 401 — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # Refresh failed — rotate to next credential instead of giving up.
            # The failed entry is already marked exhausted by try_refresh_current().
            next_entry = pool.mark_exhausted_and_rotate(status_code=401, error_context=error_context)
            if next_entry is not None:
                logger.info(f"Credential 401 (refresh failed) — rotated to pool entry {getattr(next_entry, 'id', '?')}")
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.

        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(reason="codex_stream_request")
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(reason="chat_completion_request")
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                # Force-close the in-flight worker-local HTTP connection to stop
                # token generation without poisoning the shared client used to
                # seed future retries.
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # If a tool iteration set the break flag, prepend a single paragraph
        # break before the first real text delta.  This prevents the original
        # problem (text concatenation across tool boundaries) without stacking
        # blank lines when multiple tool iterations run back-to-back.
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
        for cb in (self.stream_delta_callback, self._stream_callback):
            if cb is not None:
                try:
                    cb(text)
                except Exception:
                    pass

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        self._reasoning_deltas_fired = True
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """Notify display layer that the model is generating tool call arguments.

        Fires once per tool name when the streaming response begins producing
        tool_call / tool_use tokens.  Gives the TUI a chance to show a spinner
        or status line so the user isn't staring at a frozen screen while a
        large tool payload (e.g. a 45 KB write_file) is being generated.
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """Return True if any streaming consumer is registered."""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """Streaming variant of _interruptible_api_call for real-time token delivery.

        Handles all three api_modes:
        - chat_completions: stream=True on OpenAI-compatible endpoints
        - anthropic_messages: client.messages.stream() via Anthropic SDK
        - codex_responses: delegates to _run_codex_stream (already streaming)

        Fires stream_delta_callback and _stream_callback for each text token.
        Tool-call turns suppress the callback — only text-only final responses
        stream to the consumer.  Returns a SimpleNamespace that mimics the
        non-streaming response shape so the rest of the agent loop is unchanged.

        Falls back to _interruptible_api_call on provider errors indicating
        streaming is not supported.
        """
        if self.api_mode == "codex_responses":
            # Codex streams internally via _run_codex_stream. The main dispatch
            # in _interruptible_api_call already calls it; we just need to
            # ensure on_first_delta reaches it. Store it on the instance
            # temporarily so _run_codex_stream can pick it up.
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        result = {"response": None, "error": None}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
        # Wall-clock timestamp of the last real streaming chunk.  The outer
        # poll loop uses this to detect stale connections that keep receiving
        # SSE keep-alive pings but no actual data.
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """Stream a chat completions response."""
            import httpx as _httpx
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 60.0))
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            # Reset stale-stream timer so the detector measures from this
            # attempt's start, not a previous attempt's last chunk.
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama-compatible endpoints reuse index 0 for every tool call
            # in a parallel batch, distinguishing them only by id.  Track
            # the last seen id per raw index so we can detect a new tool
            # call starting at the same index and redirect it to a fresh slot.
            _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
            _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            # Reset per-call reasoning tracking so _build_assistant_message
            # knows whether reasoning was already displayed during streaming.
            self._reasoning_deltas_fired = False

            _first_chunk_seen = False
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                if not _first_chunk_seen:
                    _first_chunk_seen = True
                    self._touch_activity("receiving stream response")

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage comes in the final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # Accumulate reasoning content
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    else:
                        # Tool calls suppress regular content streaming (avoids
                        # displaying chatty "I'll use the tool..." text alongside
                        # tool calls).  But reasoning tags embedded in suppressed
                        # content should still reach the display — otherwise the
                        # reasoning box only appears as a post-response fallback,
                        # rendering it confusingly after the already-streamed
                        # response.  Route suppressed content through the stream
                        # delta callback so its tag extraction can fire the
                        # reasoning display.  Non-reasoning text is harmlessly
                        # suppressed by the CLI's _stream_delta when the stream
                        # box is already closed (tool boundary flush).
                        if self.stream_delta_callback:
                            try:
                                self.stream_delta_callback(delta.content)
                            except Exception:
                                pass

                # Accumulate tool call deltas — notify display on first name
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama fix: detect a new tool call reusing the same
                        # raw index (different id) and redirect to a fresh slot.
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # Fire once per tool when the full name is available
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # Build mock response matching non-streaming shape
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    ))

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=finish_reason or "stop",
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """Stream an Anthropic Messages API response.

            Fires delta callbacks for real-time token delivery, but returns
            the native Anthropic Message object from get_final_message() so
            the rest of the agent loop (validation, tool extraction, etc.)
            works unchanged.
            """
            has_tool_use = False
            self._reasoning_deltas_fired = False

            # Reset stale-stream timer for this attempt
            last_chunk_time["t"] = time.time()
            # Use the Anthropic SDK's streaming context manager
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # Return the native Anthropic Message for downstream processing
                # 获取流的最终消息
                return stream.get_final_message()

        def _call():
            # 导入 httpx 库用于处理 HTTP 连接和超时
            import httpx as _httpx

            # 从环境变量读取流式传输的最大重试次数（默认为 2）
            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                # 流式传输重试循环：最多尝试 _max_stream_retries + 1 次
                for _stream_attempt in range(_max_stream_retries + 1):
                    try:
                        # 根据不同的 API 模式调用相应的流式方法
                        if self.api_mode == "anthropic_messages":
                            # Anthropic Messages API 模式：刷新凭据后调用
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            # OpenAI Chat Completions 模式
                            result["response"] = _call_chat_completions()
                        return  # 成功完成，退出函数
                    except Exception as e:
                        # 如果已经发送了一些 token，不再重试
                        # 因为部分内容已经到达用户，重试会导致重复内容
                        if deltas_were_sent["yes"]:
                            # 流式传输在部分 token 已送达后失败
                            # 不再重试或回退 —— 部分内容已到达用户
                            logger.warning(
                                "Streaming failed after partial delivery, not retrying: %s", e
                            )
                            result["error"] = e
                            return

                        # 检测不同类型的超时错误
                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        # 检测连接错误类型
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # SSE（Server-Sent Events）代理错误检测
                        # 代理服务器（如 OpenRouter）可能发送 SSE 错误事件
                        # 例如：{"error":{"message":"Network connection lost."}}
                        # 这些错误会被 OpenAI SDK 包装为 APIError
                        # 这类错误在语义上等同于 httpx 连接断开 —— 上游流已死亡
                        # 应该使用新连接重试。区分 HTTP 错误：
                        # 来自 SSE 的 APIError 没有 status_code，而
                        # APIStatusError (4xx/5xx) 总是有 status_code。
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                # 将错误消息转为小写以便匹配
                                _err_lower_sse = str(e).lower()
                                # 定义 SSE 连接错误的常见短语列表
                                _SSE_CONN_PHRASES = (
                                    "connection lost",        # 连接丢失
                                    "connection reset",       # 连接重置
                                    "connection closed",      # 连接关闭
                                    "connection terminated",  # 连接终止
                                    "network error",          # 网络错误
                                    "network connection",     # 网络连接
                                    "terminated",             # 已终止
                                    "peer closed",            # 对端关闭
                                    "broken pipe",            # 管道破裂
                                    "upstream connect error", # 上游连接错误
                                )
                                # 检查错误消息是否包含任何连接错误短语
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        # 如果是超时、连接错误或 SSE 连接错误，进行重试
                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # 瞬时网络/超时错误。首先使用新连接重试流式请求。
                            if _stream_attempt < _max_stream_retries:
                                logger.info(
                                    "Streaming attempt %s/%s failed (%s: %s), "
                                    "retrying with fresh connection...",
                                    _stream_attempt + 1,
                                    _max_stream_retries + 1,
                                    type(e).__name__,
                                    e,
                                )
                                # 向用户显示状态消息
                                self._emit_status(
                                    f"⚠️ Connection to provider dropped "
                                    f"({type(e).__name__}). Reconnecting… "
                                    f"(attempt {_stream_attempt + 2}/{_max_stream_retries + 1})"
                                )
                                # 重试前关闭陈旧的请求客户端
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # 同时重建主客户端以清除连接池中的死连接
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                continue
                            # 所有重试都失败后显示错误消息
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                            logger.warning(
                                "Streaming exhausted %s retries on transient error, "
                                "falling back to non-streaming: %s",
                                _max_stream_retries + 1,
                                e,
                            )
                        else:
                            # 其他类型的错误（非网络错误）
                            _err_lower = str(e).lower()
                            # 检测是否为"不支持流式传输"的错误
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                # 流式传输不被此模型/提供商支持，回退到非流式
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Falling back to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery, falling back to non-streaming: %s",
                                e,
                            )

                        try:
                            # 重置陈旧计时器 —— 非流式回退使用自己的客户端
                            # 防止陈旧检测器因失败流的陈旧时间戳而触发
                            last_chunk_time["t"] = time.time()
                            # 回退到非流式 API 调用
                            result["response"] = self._interruptible_api_call(api_kwargs)
                        except Exception as fallback_err:
                            result["error"] = fallback_err
                        return
            finally:
                # 清理请求客户端资源
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        # 流式传输陈旧超时配置（默认 180 秒）
        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # 根据上下文大小调整陈旧超时时间：慢速模型（如 Opus）
        # 在大上下文情况下，可能需要几分钟才能生成第一个 token
        # 如果不调整，陈旧检测器会在模型思考阶段杀死健康连接
        # 导致虚假的 RemoteProtocolError（"对端关闭连接"）
        _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
        if _est_tokens > 100_000:
            # 超大上下文：至少 5 分钟超时
            _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
        elif _est_tokens > 50_000:
            # 大上下文：至少 4 分钟超时
            _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
        else:
            # 普通上下文：使用基础超时
            _stream_stale_timeout = _stream_stale_timeout_base

        # 在后台线程中执行流式调用
        t = threading.Thread(target=_call, daemon=True)
        t.start()
        # 主线程监控流式传输状态
        while t.is_alive():
            t.join(timeout=0.3)

            # 检测陈旧的流：连接通过 SSE ping 保持活跃
            # 但没有传递真实的块。杀死客户端以便
            # 内部重试循环可以启动新连接
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                # 计算上下文大小用于日志记录
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                # 向用户显示状态消息
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    # 获取并关闭陈旧的请求客户端
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # 同时重建主客户端 —— 其连接池可能包含
                # 来自同一提供商中断的死套接字
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # 重置计时器，避免在内部线程处理关闭时重复杀死
                last_chunk_time["t"] = time.time()

            # 处理用户中断请求
            if self._interrupt_requested:
                try:
                    # 根据不同的 API 模式处理中断
                    if self.api_mode == "anthropic_messages":
                        # Anthropic 模式：关闭并重建客户端
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        # OpenAI 模式：关闭请求客户端
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                # 抛出中断异常以停止流式传输
                raise InterruptedError("Agent interrupted during streaming API call")
        # 检查流式传输结果中的错误
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # 流式传输在部分 token 已送达平台后失败
                # 重新抛出会让外部重试循环进行新的 API 调用
                # 导致重复消息。返回部分"停止"响应以便外部循环
                # 将此轮次视为完成（无重试、无回退）
                logger.warning(
                    "Partial stream delivered before error; returning stub "
                    "response to prevent duplicate messages: %s",
                    result["error"],
                )
                # 创建一个存根消息对象
                _stub_msg = SimpleNamespace(
                    role="assistant", content=None, tool_calls=None,
                    reasoning_content=None,
                )
                # 返回一个简化的响应对象
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            # 如果没有发送过任何 token，直接抛出错误
            raise result["error"]
        # 返回成功的响应
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """Switch to the next fallback model/provider in the chain.

        Called when the current model is failing after retries.  Swaps the
        OpenAI client, model slug, and provider in-place so the retry loop
        can continue with the new backend.  Advances through the chain on
        each call; returns False when exhausted.

        Uses the centralized provider router (resolve_provider_client) for
        auth resolution and client construction — no duplicated provider→key
        mappings.
        """
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback()  # skip invalid, try next

        # Use centralized router for client construction.
        # raw_codex=True because the main agent needs direct responses.stream()
        # access for Codex providers.
        try:
            from agent.auxiliary_client import resolve_provider_client
            # Pass base_url and api_key from fallback config so custom
            # endpoints (e.g. Ollama Cloud) resolve correctly instead of
            # falling through to OpenRouter defaults.
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
            # when no explicit key is in the fallback config.
            if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _ = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True,
                explicit_base_url=fb_base_url_hint,
                explicit_api_key=fb_api_key_hint)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return self._try_activate_fallback()  # try next in chain

            # Determine api_mode from provider / base URL
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # Build native Anthropic client instead of using OpenAI client
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
                effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = getattr(fb_client, "base_url", None)
                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            else:
                # Swap OpenAI client and config in-place
                self.api_key = fb_client.api_key
                self.client = fb_client
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                }

            # Re-evaluate prompt caching for the new provider/model
            is_native_anthropic = fb_api_mode == "anthropic_messages"
            self._use_prompt_caching = (
                ("openrouter" in fb_base_url.lower() and "claude" in fb_model.lower())
                or is_native_anthropic
            )

            # Update context compressor limits for the fallback model.
            # Without this, compression decisions use the primary model's
            # context window (e.g. 200K) instead of the fallback's (e.g. 32K),
            # causing oversized sessions to overflow the fallback.
            if hasattr(self, 'context_compressor') and self.context_compressor:
                from agent.model_metadata import get_model_context_length
                fb_context_length = get_model_context_length(
                    self.model, base_url=self.base_url,
                    api_key=self.api_key, provider=self.provider,
                )
                self.context_compressor.model = self.model
                self.context_compressor.base_url = self.base_url
                self.context_compressor.api_key = self.api_key
                self.context_compressor.provider = self.provider
                self.context_compressor.context_length = fb_context_length
                self.context_compressor.threshold_tokens = int(
                    fb_context_length * self.context_compressor.threshold_percent
                )

            self._emit_status(
                f"🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback()  # try next in chain

    # ── Per-turn primary restoration ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """Restore the primary runtime at the start of a new turn.

        In long-lived CLI sessions a single AIAgent instance spans multiple
        turns.  Without restoration, one transient failure pins the session
        to the fallback provider for every subsequent turn.  Calling this at
        the top of ``run_conversation()`` makes fallback turn-scoped.

        The gateway creates a fresh agent per message so this is a no-op
        there (``_fallback_activated`` is always False at turn start).
        """
        if not self._fallback_activated:
            return False

        rt = self._primary_runtime
        try:
            # ── Core runtime state ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]           # setter updates _base_url_lower
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]

            # ── Rebuild client for the primary provider ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # ── Restore context compressor state ──
            cc = self.context_compressor
            cc.model = rt["compressor_model"]
            cc.base_url = rt["compressor_base_url"]
            cc.api_key = rt["compressor_api_key"]
            cc.provider = rt["compressor_provider"]
            cc.context_length = rt["compressor_context_length"]
            cc.threshold_tokens = rt["compressor_threshold_tokens"]

            # ── Reset fallback chain for the new turn ──
            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model, self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    # Which error types indicate a transient transport failure worth
    # one more attempt with a rebuilt client / connection pool.
    _TRANSIENT_TRANSPORT_ERRORS = frozenset({
        "ReadTimeout", "ConnectTimeout", "PoolTimeout",
        "ConnectError", "RemoteProtocolError",
    })

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """Attempt one extra primary-provider recovery cycle for transient transport failures.

        After ``max_retries`` exhaust, rebuild the primary client (clearing
        stale connection pools) and give it one more attempt before falling
        back.  This is most useful for direct endpoints (custom, Z.AI,
        Anthropic, OpenAI, local models) where a TCP-level hiccup does not
        mean the provider is down.

        Skipped for proxy/aggregator providers (OpenRouter, Nous) which
        already manage connection pools and retries server-side — if our
        retries through them are exhausted, one more rebuilt client won't help.
        """
        if self._fallback_activated:
            return False

        # Only for transient transport errors
        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        # Skip for aggregator providers — they manage their own retry infra
        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in ("nous", "nous-research"):
            return False

        try:
            # Close existing client to release stale connections
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client, reason="primary_recovery", shared=True,
                    )
                except Exception:
                    pass

            # Rebuild from primary snapshot
            rt = self._primary_runtime
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]

            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="primary_recovery",
                    shared=True,
                )

            wait_time = min(3 + retry_count, 8)
            self._vprint(
                f"{self.log_prefix}🔁 Transient {error_type} on {self.provider} — "
                f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
                force=True,
            )
            time.sleep(wait_time)
            return True
        except Exception as e:
            logging.warning("Primary transport recovery failed: %s", e)
            return False

    # ── End provider fallback ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, UI, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"Image analysis failed: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "Image analysis failed."

        note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[An image was attached but no image source was available.]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[A multimodal message was converted to text for Anthropic compatibility.]"

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _anthropic_preserve_dots(self) -> bool:
        """True when using an anthropic-compatible endpoint that preserves dots in model names.
        Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
        OpenCode Go keeps dots (e.g. minimax-m2.7)."""
        if (getattr(self, "provider", "") or "").lower() in {"alibaba", "opencode-go"}:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return "dashscope" in base or "aliyuncs" in base or "opencode.ai/zen/go" in base

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """Build the keyword arguments dict for the active API mode."""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            # Pass context_length so the adapter can clamp max_tokens if the
            # user configured a smaller context window than the model's output limit.
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
            )

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in self.base_url.lower()
                or "api.githubcopilot.com" in self.base_url.lower()
            )

            # Resolve reasoning effort: config > default (medium)
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = self.session_id

            if reasoning_enabled:
                if is_github_responses:
                    # Copilot's Responses route advertises reasoning-effort support,
                    # but not OpenAI-specific prompt cache or encrypted reasoning
                    # fields. Keep the payload to the documented subset.
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses:
                kwargs["include"] = []

            if self.max_tokens is not None:
                kwargs["max_output_tokens"] = self.max_tokens

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue

                # Codex-only replay state must not leak into strict chat-completions APIs.
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        # GPT-5 and Codex models respond better to 'developer' than 'system'
        # for instruction-following.  Swap the role at the API boundary so
        # internal message representation stays uniform ("system").
        _model_lower = (self.model or "").lower()
        if (
            sanitized_messages
            and sanitized_messages[0].get("role") == "system"
            and any(p in _model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            # Shallow-copy the list + first message only — rest stays shared.
            sanitized_messages = list(sanitized_messages)
            sanitized_messages[0] = {**sanitized_messages[0], "role": "developer"}

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 1800.0)),
        }
        if self.tools:
            api_kwargs["tools"] = self.tools

        if self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))
        elif self._is_openrouter_url() and "claude" in (self.model or "").lower():
            # OpenRouter translates requests to Anthropic's Messages API,
            # which requires max_tokens as a mandatory field.  When we omit
            # it, OpenRouter picks a default that can be too low — the model
            # spends its output budget on thinking and has almost nothing
            # left for the actual response (especially large tool calls like
            # write_file).  Sending the model's real output limit ensures
            # full capacity.  Other providers handle the default fine.
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _model_output_limit = _get_anthropic_max_output(self.model)
                api_kwargs["max_tokens"] = _model_output_limit
            except Exception:
                pass  # fail open — let OpenRouter pick its default

        extra_body = {}

        _is_openrouter = self._is_openrouter_url()
        _is_github_models = (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        )

        # Provider preferences (only, ignore, order, sort) are OpenRouter-
        # specific.  Only send to OpenRouter-compatible endpoints.
        # TODO: Nous Portal will add transparent proxy support — re-enable
        # for _is_nous when their backend is updated.
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self._base_url_lower

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if self.reasoning_config is not None:
                    rc = dict(self.reasoning_config)
                    # Nous Portal requires reasoning enabled — don't send
                    # enabled=false to it (would cause 400).
                    if _is_nous and rc.get("enabled") is False:
                        pass  # omit reasoning entirely for Nous when disabled
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }

        # Nous Portal product attribution
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # xAI prompt caching: send x-grok-conv-id header to route requests
        # to the same server, maximizing automatic cache hits.
        # https://docs.x.ai/developers/advanced-api-usage/prompt-caching
        if "x.ai" in self._base_url_lower and hasattr(self, "session_id") and self.session_id:
            api_kwargs["extra_headers"] = {"x-grok-conv-id": self.session_id}

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if "nousresearch" in self._base_url_lower:
            return True
        if "ai-gateway.vercel.sh" in self._base_url_lower:
            return True
        if "models.github.ai" in self._base_url_lower or "api.githubcopilot.com" in self._base_url_lower:
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # displayed during the stream via one of two paths:
            #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
            #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
            # When streaming is NOT active, always fire so non-streaming modes
            # (gateway, batch, quiet) still get reasoning.
            # Any reasoning that wasn't shown during streaming is caught by the
            # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
            if not self.stream_delta_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # Pass reasoning_details back unmodified so providers (OpenRouter,
            # Anthropic, OpenAI) can maintain reasoning continuity across turns.
            # Each provider may include opaque fields (signature, encrypted_content)
            # that must be preserved exactly.
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API: preserve encrypted reasoning items for
        # multi-turn continuity. These get replayed as input on the next turn.
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # Preserve extra_content (e.g. Gemini thought_signature) so it
                # is sent back on subsequent API calls.  Without this, Gemini 3
                # thinking models reject the request with a 400 error.
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    def _should_sanitize_tool_calls(self) -> bool:
        """Determine if tool_calls need sanitization for strict APIs.

        Codex Responses API uses fields like call_id and response_item_id
        that are not part of the standard Chat Completions schema. These
        fields must be stripped when calling any other API to avoid
        validation errors (400 Bad Request).

        Returns:
            bool: True if sanitization is needed (non-Codex API), False otherwise.
        """
        return self.api_mode != "codex_responses"

    def flush_memories(self, messages: list = None, min_turns: int = None):
        """Give the model one turn to persist memories before context is lost.

        Called before compression, session reset, or CLI exit. Injects a flush
        message, makes one API call, executes any memory tool calls, then
        strips all flush artifacts from the message list.

        Args:
            messages: The current conversation messages. If None, uses
                      self._session_messages (last run_conversation state).
            min_turns: Minimum user turns required to trigger the flush.
                       None = use config value (flush_min_turns).
                       0 = always flush (used for compression).
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            # Build API messages for the flush call
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            # Make one API call with only the memory tool available
            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # Use auxiliary client for the flush call when available --
            # it's cheaper and avoids Codex Responses API incompatibility.
            from agent.auxiliary_client import call_llm as _call_llm
            _aux_available = True
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=0.3,
                    max_tokens=5120,
                    timeout=30.0,
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # No auxiliary client -- use the Codex Responses path directly
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = 0.3
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # Native Anthropic — use the Anthropic client directly
                from agent.anthropic_adapter import build_anthropic_kwargs as _build_ant_kwargs
                ant_kwargs = _build_ant_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": 0.3,
                    **self._max_tokens_param(5120),
                }
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(**api_kwargs, timeout=30.0)

            # Extract tool calls from the response, handling all API formats
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import normalize_anthropic_response as _nar_flush
                _flush_msg, _ = _nar_flush(response, strip_tool_prefix=self._is_anthropic_oauth)
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif hasattr(response, "choices") and response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        result = _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # Strip flush artifacts: remove everything from the flush message onward.
            # Use sentinel marker instead of identity check for robustness.
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default") -> tuple:
        """Compress conversation context and split the session in SQLite.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
        )
        # Pre-compression memory flush: let the model save memories before they're lost
        self.flush_memories(messages, min_turns=0)

        # Notify external memory provider before compression discards context
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                # Update session_log_file to point to the new session's JSON file
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                # Auto-number the title for the continuation session
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Update token estimate after compaction so pressure calculations
        # use the post-compression count, not the stale pre-compression one.
        _compressed_est = (
            estimate_tokens_rough(new_system_prompt)
            + estimate_messages_tokens_rough(compressed)
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # Only reset the pressure warning if compression actually brought
        # us below the warning level (85% of threshold).  When compression
        # can't reduce enough (e.g. threshold is very low, or system prompt
        # alone exceeds the warning level), keep the flag set to prevent
        # spamming the user with repeated warnings every loop iteration.
        if self.context_compressor.threshold_tokens > 0:
            _post_progress = _compressed_est / self.context_compressor.threshold_tokens
            if _post_progress < 0.85:
                self._context_pressure_warned = False

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        Dispatches to concurrent execution only for batches that look
        independent: read-only tools may always share the parallel path, while
        file reads/writes may do so only when their target paths do not overlap.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
            )

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag)
        results = [None] * num_tools

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id, tool_call.id)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error)

        # Start spinner for CLI mode (skip when TUI handles tool progress)
        spinner = None
        if self.quiet_mode and not self.tool_progress_callback and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # Wait for all to complete (exceptions are captured inside _run_tool)
                concurrent.futures.wait(futures)
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # Shouldn't happen, but safety fallback
                function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self.quiet_mode:
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            # Save oversized results to file instead of destructive truncation
            function_result = _save_oversized_tool_result(name, function_result)

            # Discover subdirectory context files from tool arguments
            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                function_result += subdir_hints

            # Append tool result message in order
            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── Budget pressure injection ────────────────────────────────────
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️  WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # SAFETY: check interrupt BEFORE starting each tool.
            # If the user sent "stop" during a previous tool's execution,
            # do NOT start any more tools -- skip them all immediately.
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            # Reset nudge counters when the relevant tool is actually used
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            self._current_tool = function_name
            self._touch_activity(f"executing tool: {function_name}")

            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            # 处理待办事项工具调用
            # 该工具用于管理待办事项列表，支持添加、更新、删除和合并待办事项
            if function_name == "todo":
                # 动态导入待办事项工具模块，避免循环导入
                from tools.todo_tool import todo_tool as _todo_tool
                # 调用待办事项工具，传入待办事项列表、合并标志和存储实例
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                # 计算工具执行耗时
                tool_duration = time.time() - tool_start_time
                # 如果处于安静模式，输出格式化的工具执行信息
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            # 处理会话搜索工具调用
            # 该工具用于在当前会话历史中搜索相关内容，支持按角色过滤和限制结果数量
            elif function_name == "session_search":
                # 检查会话数据库是否可用，如果不可用返回错误信息
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    # 动态导入会话搜索工具模块
                    from tools.session_search_tool import session_search as _session_search
                    # 调用会话搜索工具，传入搜索查询、角色过滤器、结果限制数量、数据库实例和当前会话ID
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                # 计算工具执行耗时
                tool_duration = time.time() - tool_start_time
                # 如果处于安静模式，输出格式化的工具执行信息
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            # 处理内存工具调用
            # 该工具用于管理智能体的记忆系统，支持添加、搜索、更新和删除记忆内容
            elif function_name == "memory":
                # 获取目标存储类型，默认为"memory"
                target = function_args.get("target", "memory")
                # 动态导入内存工具模块
                from tools.memory_tool import memory_tool as _memory_tool
                # 调用内存工具，传入操作类型、目标、内容、旧文本和存储实例
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                # 计算工具执行耗时
                tool_duration = time.time() - tool_start_time
                # 如果处于安静模式，输出格式化的工具执行信息
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            # 处理澄清工具调用
            # 该工具用于向用户请求澄清或确认，支持选择题和开放式问题
            elif function_name == "clarify":
                # 动态导入澄清工具模块
                from tools.clarify_tool import clarify_tool as _clarify_tool
                # 调用澄清工具，传入问题、选项列表和回调函数
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                # 计算工具执行耗时
                tool_duration = time.time() - tool_start_time
                # 如果处于安静模式，输出格式化的工具执行信息
                if self.quiet_mode:
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            # 处理任务委托工具调用
            # 该工具用于将复杂任务分解为子任务并委托给其他智能体处理，支持批量任务处理
            elif function_name == "delegate_task":
                # 动态导入任务委托工具模块
                from tools.delegate_tool import delegate_task as _delegate_task
                # 获取任务参数，判断是否为批量任务
                tasks_arg = function_args.get("tasks")
                # 根据任务类型生成不同的加载提示信息
                if tasks_arg and isinstance(tasks_arg, list):
                    # 如果是任务列表，显示委托任务数量
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    # 如果是单个目标，显示目标预览（前30个字符）
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                # 初始化加载动画控制器
                spinner = None
                # 如果处于安静模式且没有工具进度回调，启动可爱的加载动画
                if self.quiet_mode and not self.tool_progress_callback and self._should_start_quiet_spinner():
                    # 随机选择一个可爱的等待表情
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    # 创建加载动画实例
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    # 启动加载动画
                    spinner.start()
                # 保存委托加载动画引用，用于后续停止
                self._delegate_spinner = spinner
                # 初始化委托结果变量
                _delegate_result = None
                try:
                    # 调用任务委托工具，传入目标、上下文、工具集、任务列表、最大迭代次数和父智能体引用
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    # 保存委托结果
                    _delegate_result = function_result
                finally:
                    # 清理委托加载动画引用
                    self._delegate_spinner = None
                    # 计算工具执行耗时
                    tool_duration = time.time() - tool_start_time
                    # 生成格式化的工具执行信息
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    # 停止加载动画或输出执行信息
                    if spinner:
                        # 如果有加载动画，用执行信息停止动画
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        # 否则直接输出执行信息
                        self._vprint(f"  {cute_msg}")
            # 处理内存管理器提供的工具调用
            # 这些工具不在常规工具注册表中，而是通过内存管理器路由处理
            # 包括 hindsight_retain、honcho_search 等内存相关工具
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # 内存提供工具（hindsight_retain、honcho_search 等）
                # 这些工具不在工具注册表中 —— 通过 MemoryManager 路由处理
                # 初始化加载动画控制器
                spinner = None
                # 如果处于安静模式且没有工具进度回调，启动可爱的加载动画
                if self.quiet_mode and not self.tool_progress_callback:
                    # 随机选择一个可爱的等待表情
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    # 获取工具的emoji图标
                    emoji = _get_tool_emoji(function_name)
                    # 构建工具预览信息，如果无法构建则使用工具名称
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    # 创建加载动画实例，包含表情、图标和预览信息
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    # 启动加载动画
                    spinner.start()
                # 初始化内存工具结果变量
                _mem_result = None
                try:
                    # 通过内存管理器处理工具调用，传入工具名称和参数
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    # 保存内存工具执行结果
                    _mem_result = function_result
                except Exception as tool_error:
                    # 如果内存工具执行失败，生成错误信息并记录日志
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    # 计算工具执行耗时
                    tool_duration = time.time() - tool_start_time
                    # 生成格式化的工具执行信息
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    # 停止加载动画或输出执行信息
                    if spinner:
                        # 如果有加载动画，用执行信息停止动画
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        # 否则直接输出执行信息
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if not self.tool_progress_callback:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    else:
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, len(function_result))

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            # Save oversized results to file instead of destructive truncation
            function_result = _save_oversized_tool_result(function_name, function_result)

            # Discover subdirectory context files from tool arguments
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Budget pressure injection ─────────────────────────────────
        # After all tool calls in this turn are processed, check if we're
        # approaching max_iterations. If so, inject a warning into the LAST
        # tool result's JSON so the LLM sees it naturally when reading results.
        budget_warning = self._get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = "⚠️  WARNING" if remaining <= self.max_iterations * 0.1 else "💡 CAUTION"
                print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

    def _get_budget_warning(self, api_call_count: int) -> Optional[str]:
        """Return a budget pressure string, or None if not yet needed.

        Two-tier system:
          - Caution (70%): nudge to consolidate work
          - Warning (90%): urgent, must respond now
        """
        if not self._budget_pressure_enabled or self.max_iterations <= 0:
            return None
        progress = api_call_count / self.max_iterations
        remaining = self.max_iterations - api_call_count
        if progress >= self._budget_warning_threshold:
            return (
                f"[BUDGET WARNING: Iteration {api_call_count}/{self.max_iterations}. "
                f"Only {remaining} iteration(s) left. "
                "Provide your final response NOW. No more tool calls unless absolutely critical.]"
            )
        if progress >= self._budget_caution_threshold:
            return (
                f"[BUDGET: Iteration {api_call_count}/{self.max_iterations}. "
                f"{remaining} iterations left. Start consolidating your work.]"
            )
        return None

    def _emit_context_pressure(self, compaction_progress: float, compressor) -> None:
        """Notify the user that context is approaching the compaction threshold.

        Args:
            compaction_progress: How close to compaction (0.0–1.0, where 1.0 = fires).
            compressor: The ContextCompressor instance (for threshold/context info).

        Purely user-facing — does NOT modify the message stream.
        For CLI: prints a formatted line with a progress bar.
        For gateway: fires status_callback so the platform can send a chat message.
        """
        from agent.display import format_context_pressure, format_context_pressure_gateway

        threshold_pct = compressor.threshold_tokens / compressor.context_length if compressor.context_length else 0.5

        # CLI output — always shown (these are user-facing status notifications,
        # not verbose debug output, so they bypass quiet_mode).
        # Gateway users also get the callback below.
        if self.platform in (None, "cli"):
            line = format_context_pressure(
                compaction_progress=compaction_progress,
                threshold_tokens=compressor.threshold_tokens,
                threshold_percent=threshold_pct,
                compression_enabled=self.compression_enabled,
            )
            self._safe_print(line)

        # Gateway / external consumers
        if self.status_callback:
            try:
                msg = format_context_pressure_gateway(
                    compaction_progress=compaction_progress,
                    threshold_percent=threshold_pct,
                    compression_enabled=self.compression_enabled,
                )
                self.status_callback("context_pressure", msg)
            except Exception:
                logger.debug("status_callback error in context pressure", exc_info=True)

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Request a summary when max iterations are reached. Returns the final response text."""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # Build API messages, stripping internal-only fields
            # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (assistant_message.content or "").strip() if assistant_message else ""
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak, normalize_anthropic_response as _nar
                    _ant_kw = _bak(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # Retry summary generation
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (retry_msg.content or "").strip() if retry_msg else ""
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak2, normalize_anthropic_response as _nar2
                    _ant_kw2 = _bak2(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的对话循环，包含工具调用，直到任务完成。

        这是 AIAgent 的核心方法，负责处理完整的对话流程：
        从用户消息开始，通过多轮工具调用，直到产生最终响应。
        该方法实现了完整的对话生命周期管理，包括上下文管理、
        工具执行、错误恢复和会话持久化。

        工作流程：
        1. 输入验证和清理：处理代理字符、生成任务 ID 等
        2. 会话初始化：加载历史消息、系统提示词
        3. 上下文检查：确保不会超过模型上下文限制
        4. 主循环：调用 LLM → 执行工具 → 返回结果，重复直到完成
        5. 后处理：保存会话、触发后台审查（记忆/技能）

        Args:
            user_message (str): 用户的消息或问题，这是对话的输入
            system_message (str): 自定义系统消息（可选，如果提供则覆盖 ephemeral_system_prompt）
            conversation_history (List[Dict]): 之前的对话消息历史（可选，用于继续会话）
            task_id (str): 此任务的唯一标识符，用于隔离并发任务之间的虚拟机
                          （可选，如果未提供则自动生成）
            stream_callback: 可选的回调函数，在流式传输期间随每个文本增量调用。
                           由 TTS 管道用于在完整响应之前开始音频生成。
                           当为 None（默认）时，API 调用使用标准的非流式路径。
            persist_user_message: 当 user_message 包含仅 API 的合成前缀时，
                                要存储在转录/历史中的干净用户消息（可选）。

        Returns:
            Dict: 包含最终响应和消息历史的完整对话结果
                 - final_response: 最终的文本响应
                 - messages: 完整的消息历史
                 - api_calls: API 调用次数
                 - completed: 是否成功完成
                 - error: 错误信息（如果有）
        """
        # 防止管道破裂导致的 OSError（systemd/无头/守护进程模式）
        # 只安装一次，当流健康时透明，防止写入时崩溃
        _install_safe_stdio()

        # 如果上一轮激活了回退模型，恢复主运行时，
        # 这样这一轮可以用首选模型重新尝试。
        # 当 _fallback_activated 为 False 时为空操作（网关、第一轮等）。
        self._restore_primary_runtime()

        # 清理用户输入中的代理字符。
        # 从富文本编辑器（Google Docs、Word 等）粘贴的内容可能注入单独的代理字符，
        # 这些在 UTF-8 中是无效的，会导致 OpenAI SDK 中的 JSON 序列化崩溃。
        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        # 存储流回调供 _interruptible_api_call 使用
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message

        # 如果未提供，生成唯一的 task_id 以隔离并发任务之间的虚拟机
        effective_task_id = task_id or str(uuid.uuid4())

        # 在每轮开始时重置重试计数器和迭代预算，
        # 这样上一轮的子智能体使用不会影响下一轮。
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._last_content_with_tools = None
        self._mute_post_response = False
        self._surrogate_sanitized = False

        # 轮次前连接健康检查：检测并清理死 TCP 连接，
        # 这些连接是从提供商中断或丢弃的流中遗留下来的。
        # 这防止下一个 API 调用挂在僵尸套接字上。
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 Detected stale connections from a previous provider "
                        "issue — cleaned up automatically. Proceeding with fresh "
                        "connection."
                    )
            except Exception:
                pass

        # 注意：_turns_since_memory 和 _iters_since_skill 在这里不重置。
        # 它们在 __init__ 中初始化，必须在 run_conversation 调用之间持久化，
        # 以便在 CLI 模式下正确累积提示逻辑。
        self.iteration_budget = IterationBudget(self.max_iterations)

        # 记录对话轮次开始，用于调试/可观察性
        _msg_preview = (user_message[:80] + "...") if len(user_message) > 80 else user_message
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # 初始化对话（复制以避免修改调用者的列表）
        messages = list(conversation_history) if conversation_history else []

        # 从之前的轮次中剥离预算压力警告。
        # 这些是由 _get_budget_warning() 注入到工具结果内容中的轮次范围信号。
        # 如果保留在重放的历史中，模型（特别是 GPT 系列）会将它们解释为仍然活跃的指令，
        # 并避免在所有后续轮次中进行工具调用。
        if messages:
            _strip_budget_warnings_from_history(messages)

        # 从对话历史中补充 todo 存储（网关为每条消息创建一个新的 AIAgent，
        # 所以内存存储是空的 —— 我们需要从历史中最近的 todo 工具响应恢复 todo 状态）
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
        # 预填充消息（few-shot 引导）仅在 API 调用时注入，
        # 从不存储在消息列表中。这使它们保持短暂：
        # 它们不会保存到会话数据库、会话日志或批量轨迹，
        # 但会在每次 API 调用时自动重新应用（包括会话继续）。

        # 跟踪用户轮次用于记忆刷新和定期提示逻辑
        self._user_turn_count += 1

        # 保留原始用户消息（无提示注入）。
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # 跟踪记忆提示触发器（基于轮次，在此检查）。
        # 技能触发器在智能体循环完成后检查，基于此轮使用的工具迭代次数。
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # 添加用户消息
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx

        if not self.quiet_mode:
            self._safe_print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # ── 系统提示词（每个会话缓存用于前缀缓存）──
        # 在第一次调用时构建一次，在所有后续调用中重用。
        # 仅在上下文压缩事件后重建（这会使缓存失效并从磁盘重新加载记忆）。
        #
        # 对于继续的会话（网关为每条消息创建一个新的 AIAgent），
        # 我们从会话数据库加载存储的系统提示词而不是重建。
        # 重建会从磁盘获取模型已经知道的记忆更改（它写入的！），
        # 产生不同的系统提示词并破坏 Anthropic 前缀缓存。
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # 回退到从头构建

            if stored_prompt:
                # 继续会话 —— 重用上一轮的确切系统提示词，
                # 以便 Anthropic 缓存前缀匹配。
                self._cached_system_prompt = stored_prompt
            else:
                # 新会话的第一轮 —— 从头构建。
                self._cached_system_prompt = self._build_system_prompt(system_message)

                # 插件钩子：on_session_start
                # 在创建全新会话时触发一次（不在继续时）。
                # 插件可以使用此钩子初始化会话范围的状态（例如预热记忆缓存）。
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                # 在 SQLite 中存储系统提示词快照
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt

        # ── Preflight context compression ──
        # Before entering the main loop, check if the loaded conversation
        # history already exceeds the model's context threshold.  This handles
        # cases where a user switches to a model with a smaller context window
        # while having a large existing session — compress proactively rather
        # than waiting for an API error (which might be caught as a non-retryable
        # 4xx and abort the request entirely).
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            # Include tool schema tokens — with many tools these can add
            # 20-30K+ tokens that the old sys+msg estimate missed entirely.
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 Preflight compression: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} threshold"
                    )
                # May need multiple passes for very large sessions with small
                # context windows (each pass summarises the middle N turns).
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # Cannot compress further
                    # Compression created a new session — clear the history
                    # reference so _flush_messages_to_session_db writes ALL
                    # compressed messages to the new session's SQLite, not
                    # skipping them because conversation_history is still the
                    # pre-compression length.
                    conversation_history = None
                    # Re-estimate after compression
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # Under threshold

        # Plugin hook: pre_llm_call
        # Fired once per turn before the tool-calling loop.  Plugins can
        # return a dict with a ``context`` key (or a plain string) whose
        # value is appended to the current turn's user message.
        #
        # Context is ALWAYS injected into the user message, never the
        # system prompt.  This preserves the prompt cache prefix — the
        # system prompt stays identical across turns so cached tokens
        # are reused.  The system prompt is Hermes's territory; plugins
        # contribute context alongside the user's input.
        #
        # All injected context is ephemeral (not persisted to session DB).
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        # Main conversation loop
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        
        # Clear any stale interrupt state at start
        self.clear_interrupt()

        # External memory provider: prefetch once before the tool loop.
        # Reuse the cached result on every iteration to avoid re-calling
        # prefetch_all() on each tool call (10 tool calls = 10x latency + cost).
        # Use original_user_message (clean input) — user_message may contain
        # injected skill content that bloats / breaks provider queries.
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
            except Exception:
                pass

        while api_call_count < self.max_iterations and self.iteration_budget.remaining > 0:
            # Reset per-turn checkpoint dedup so each iteration can take one snapshot
            self._checkpoint_mgr.new_turn()

            # Check for interrupt request (e.g., user sent new message)
            if self._interrupt_requested:
                interrupted = True
                if not self.quiet_mode:
                    self._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"starting API call #{api_call_count}")
            if not self.iteration_budget.consume():
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
                break

            # Fire step_callback for gateway hooks (agent:step event)
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # Track tool-calling iterations for skill nudge.
            # Counter resets whenever skill_manage is actually used.
            if (self._skill_nudge_interval > 0
                    and "skill_manage" in self.valid_tool_names):
                self._iters_since_skill += 1
            
            # Prepare messages for API call
            # If we have an ephemeral system prompt, prepend it to the messages
            # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
            # However, providers like Moonshot AI require a separate 'reasoning_content' field
            # on assistant messages with tool_calls. We handle both cases here.
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                # Inject ephemeral context into the current turn's user message.
                # Sources: memory manager prefetch + plugin pre_llm_call hooks
                # with target="user_message" (the default).  Both are
                # API-call-time only — the original message in `messages` is
                # never mutated, so nothing leaks into session persistence.
                if idx == current_turn_user_idx and msg.get("role") == "user":
                    _injections = []
                    if _ext_prefetch_cache:
                        _fenced = build_memory_context_block(_ext_prefetch_cache)
                        if _fenced:
                            _injections.append(_fenced)
                    if _plugin_user_context:
                        _injections.append(_plugin_user_context)
                    if _injections:
                        _base = api_msg.get("content", "")
                        if isinstance(_base, str):
                            api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

                # For ALL assistant messages, pass reasoning back to the API
                # This ensures multi-turn reasoning context is preserved
                if msg.get("role") == "assistant":
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # Add reasoning_content for API compatibility (Moonshot AI, Novita, OpenRouter)
                        api_msg["reasoning_content"] = reasoning_text

                # Remove 'reasoning' field - it's for trajectory storage only
                # We've copied it to 'reasoning_content' for the API above
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # Strip Codex Responses API fields (call_id, response_item_id) for
                # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
                # Uses new dicts so the internal messages list retains the fields
                # for Codex Responses compatibility.
                if self._should_sanitize_tool_calls():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
                # The signature field helps maintain reasoning continuity
                api_messages.append(api_msg)

            # Build the final system message: cached prompt + ephemeral system prompt.
            # Ephemeral additions are API-call-time only (not persisted to session DB).
            # External recall context is injected into the user message, not the system
            # prompt, so the stable cache prefix remains unchanged.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            # NOTE: Plugin context from pre_llm_call hooks is injected into the
            # user message (see injection block above), NOT the system prompt.
            # This is intentional — system prompt modifications break the prompt
            # cache prefix.  The system prompt is reserved for Hermes internals.
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # Inject ephemeral prefill messages right after the system prompt
            # but before conversation history. Same API-call-time-only pattern.
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Apply Anthropic prompt caching for Claude models via OpenRouter.
            # Auto-detected: if model name contains "claude" and base_url is OpenRouter,
            # inject cache_control breakpoints (system + last 3 messages) to reduce
            # input token costs by ~75% on multi-turn conversations.
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=self._cache_ttl, native_anthropic=(self.api_mode == 'anthropic_messages'))

            # Safety net: strip orphaned tool results / add stubs for missing
            # results before sending to the API.  Runs unconditionally — not
            # gated on context_compressor — so orphans from session loading or
            # manual message manipulation are always caught.
            api_messages = self._sanitize_api_messages(api_messages)

            # Calculate approximate request size for logging
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4  # Rough estimate: 4 chars per token
            
            # Thinking spinner for quiet mode (animated during API call)
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
                self._vprint(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            else:
                # Animated thinking spinner in quiet mode
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
                if self.thinking_callback:
                    # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                    # (works in both streaming and non-streaming modes)
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # Raw KawaiiSpinner only when no streaming consumers and the
                    # spinner output has a safe sink.
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()
            
            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted=False
            anthropic_auth_retry_attempted=False
            nous_auth_retry_attempted=False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # Guard against UnboundLocalError if all retries fail

            while retry_count < max_retries:
                try:
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                    try:
                        from hermes_cli.plugins import invoke_hook as _invoke_hook
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            session_id=self.session_id or "",
                            platform=self.platform or "",
                            model=self.model,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_mode=self.api_mode,
                            api_call_count=api_call_count,
                            message_count=len(api_messages),
                            tool_count=len(self.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=self.max_tokens,
                        )
                    except Exception:
                        pass

                    if env_var_enabled("HERMES_DUMP_REQUESTS"):
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # Always prefer the streaming path — even without stream
                    # consumers.  Streaming gives us fine-grained health
                    # checking (90s stale-stream detection, 60s read timeout)
                    # that the non-streaming path lacks.  Without this,
                    # subagents and other quiet-mode callers can hang
                    # indefinitely when the provider keeps the connection
                    # alive with SSE pings but never delivers a response.
                    # The streaming path is a no-op for callbacks when no
                    # consumers are registered, and falls back to non-
                    # streaming automatically if the provider doesn't
                    # support it.
                    def _stop_spinner():
                        nonlocal thinking_spinner
                        if thinking_spinner:
                            thinking_spinner.stop("")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                    _use_streaming = True
                    if not self._has_stream_consumers():
                        # No display/TTS consumer. Still prefer streaming for
                        # health checking, but skip for Mock clients in tests
                        # (mocks return SimpleNamespace, not stream iterators).
                        from unittest.mock import Mock
                        if isinstance(getattr(self, "client", None), Mock):
                            _use_streaming = False

                    if _use_streaming:
                        response = self._interruptible_streaming_api_call(
                            api_kwargs, on_first_delta=_stop_spinner
                        )
                    else:
                        response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # Stop thinking spinner silently -- the response box or tool
                    # execution messages that follow are more informative.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # Log response with provider info if available
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # Validate response shape before proceeding
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        output_items = getattr(response, "output", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(output_items, list):
                            response_invalid = True
                            error_details.append("response.output is not a list")
                        elif len(output_items) == 0:
                            # If we reach here, _run_codex_stream's backfill
                            # from output_item.done events and text-delta
                            # synthesis both failed to populate output.
                            _resp_status = getattr(response, "status", None)
                            _resp_incomplete = getattr(response, "incomplete_details", None)
                            logging.warning(
                                "Codex response.output is empty after stream backfill "
                                "(status=%s, incomplete_details=%s, model=%s). %s",
                                _resp_status, _resp_incomplete,
                                getattr(response, "model", None),
                                f"api_mode={self.api_mode} provider={self.provider}",
                            )
                            response_invalid = True
                            error_details.append("response.output is empty")
                    elif self.api_mode == "anthropic_messages":
                        content_blocks = getattr(response, "content", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(content_blocks, list):
                            response_invalid = True
                            error_details.append("response.content is not a list")
                        elif len(content_blocks) == 0:
                            response_invalid = True
                            error_details.append("response.content is empty")
                    else:
                        if response is None or not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, 'choices'):
                                error_details.append("response has no 'choices' attribute")
                            elif response.choices is None:
                                error_details.append("response.choices is None")
                            else:
                                error_details.append("response.choices is empty")

                    if response_invalid:
                        # Stop spinner before printing error messages
                        if thinking_spinner:
                            thinking_spinner.stop("(´;ω;`) oops, retrying...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # This is often rate limiting or provider returning malformed response
                        retry_count += 1
                        
                        # Eager fallback: empty/malformed responses are a common
                        # rate-limit symptom.  Switch to fallback immediately
                        # rather than retrying with extended backoff.
                        if self._fallback_index < len(self._fallback_chain):
                            self._emit_status("⚠️ Empty/malformed response — switching to fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue

                        # Check for error field in response (some providers include this)
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # Try to extract provider from error metadata
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # Try to get provider from model field (OpenRouter often returns actual model used)
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        # Check for x-openrouter-provider or similar metadata
                        if provider_name == "Unknown" and response:
                            # Log all response attributes for debugging
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                        
                        self._vprint(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 Provider: {provider_name}", force=True)
                        cleaned_provider_error = self._clean_error_message(error_msg)
                        self._vprint(f"{self.log_prefix}   📝 Provider message: {cleaned_provider_error}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  Response time: {api_duration:.2f}s (fast response often indicates rate limiting)", force=True)
                        
                        if retry_count >= max_retries:
                            # Try fallback before giving up
                            self._emit_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                continue
                            self._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Invalid API response shape. Likely rate limited or malformed provider response.",
                                "failed": True  # Mark as failure for filtering
                            }
                        
                        # Longer backoff for rate limiting (likely cause of None choices)
                        wait_time = min(5 * (2 ** (retry_count - 1)), 120)  # 5s, 10s, 20s, 40s, 80s, 120s
                        self._vprint(f"{self.log_prefix}⏳ Retrying in {wait_time}s (extended backoff for possible rate limit)...", force=True)
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        
                        # Sleep in small increments to stay responsive to interrupts
                        sleep_end = time.time() + wait_time
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"Operation interrupted: retrying API call after rate limit (retry {retry_count}/{max_retries}).",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                        continue  # Retry the API call

                    # Check finish_reason before proceeding
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(response, "incomplete_details", None)
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(incomplete_details, "reason", None)
                        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                        finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                    else:
                        finish_reason = response.choices[0].finish_reason

                    if finish_reason == "length":
                        self._vprint(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens", force=True)

                        # ── Detect thinking-budget exhaustion ──────────────
                        # When the model spends ALL output tokens on reasoning
                        # and has none left for the response, continuation
                        # retries are pointless.  Detect this early and give a
                        # targeted error instead of wasting 3 API calls.
                        _trunc_content = None
                        if self.api_mode == "chat_completions":
                            _trunc_msg = response.choices[0].message if (hasattr(response, "choices") and response.choices) else None
                            _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                        elif self.api_mode == "anthropic_messages":
                            # Anthropic response.content is a list of blocks
                            _text_parts = []
                            for _blk in getattr(response, "content", []):
                                if getattr(_blk, "type", None) == "text":
                                    _text_parts.append(getattr(_blk, "text", ""))
                            _trunc_content = "\n".join(_text_parts) if _text_parts else None

                        _thinking_exhausted = (
                            _trunc_content is not None
                            and not self._has_content_after_think_block(_trunc_content)
                        ) or _trunc_content is None

                        if _thinking_exhausted:
                            _exhaust_error = (
                                "Model used all output tokens on reasoning with none left "
                                "for the response. Try lowering reasoning effort or "
                                "increasing max_tokens."
                            )
                            self._vprint(
                                f"{self.log_prefix}💭 Reasoning exhausted the output token budget — "
                                f"no visible response was produced.",
                                force=True,
                            )
                            # Return a user-friendly message as the response so
                            # CLI (response box) and gateway (chat message) both
                            # display it naturally instead of a suppressed error.
                            _exhaust_response = (
                                "⚠️ **Thinking Budget Exhausted**\n\n"
                                "The model used all its output tokens on reasoning "
                                "and had none left for the actual response.\n\n"
                                "To fix this:\n"
                                "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                                "→ Increase the output token limit: "
                                "set `model.max_tokens` in config.yaml"
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": _exhaust_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _exhaust_error,
                            }

                        if self.api_mode == "chat_completions":
                            assistant_message = response.choices[0].message
                            if not assistant_message.tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += assistant_message.content

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/3)..."
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: Your previous response was truncated by the output "
                                            "length limit. Continue exactly where you left off. Do not "
                                            "restart or repeat prior text. Finish the answer directly.]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "Response remained truncated after 3 continuation attempts",
                                }

                        # If we have prior messages, roll back to last complete state
                        if len(messages) > 1:
                            self._vprint(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit"
                            }
                        else:
                            # First message was truncated - mark as failed
                            self._vprint(f"{self.log_prefix}❌ First response truncated - cannot recover", force=True)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit"
                            }
                    
                    # Track actual token usage from response for context management
                    if hasattr(response, 'usage') and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # Cache discovered context length after successful call.
                        # Only persist limits confirmed by the provider (parsed
                        # from the error message), not guessed probe tiers.
                        if self.context_compressor._context_probed:
                            ctx = self.context_compressor.context_length
                            if getattr(self.context_compressor, "_context_probe_persistable", False):
                                save_context_length(self.model, self.base_url, ctx)
                                self._safe_print(f"{self.log_prefix}💾 Cached context length: {ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False
                            self.context_compressor._context_probe_persistable = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        # Log API call details for debugging/observability
                        _cache_pct = ""
                        if canonical_usage.cache_read_tokens and prompt_tokens:
                            _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                        logger.info(
                            "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                            self.session_api_calls, self.model, self.provider or "unknown",
                            prompt_tokens, completion_tokens, total_tokens,
                            api_duration, _cache_pct,
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # Persist token counts to session DB for /insights.
                        # Do this for every platform with a session_id so non-CLI
                        # sessions (gateway, cron, delegated runs) cannot lose
                        # token/accounting data if a higher-level persistence path
                        # is skipped or fails. Gateway/session-store writes use
                        # absolute totals, so they safely overwrite these per-call
                        # deltas instead of double-counting them.
                        if self._session_db and self.session_id:
                            try:
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=float(cost_result.amount_usd)
                                    if cost_result.amount_usd is not None else None,
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode="subscription_included"
                                    if cost_result.status == "included" else None,
                                    model=self.model,
                                )
                            except Exception:
                                pass  # never block the agent loop
                        
                        if self.verbose_logging:
                            logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # Log cache hit stats when prompt caching is active
                        if self._use_prompt_caching:
                            if self.api_mode == "anthropic_messages":
                                # Anthropic uses cache_read_input_tokens / cache_creation_input_tokens
                                cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                                written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                            else:
                                # OpenRouter uses prompt_tokens_details.cached_tokens
                                details = getattr(response.usage, 'prompt_tokens_details', None)
                                cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                                written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                self._vprint(f"{self.log_prefix}   💾 Cache: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% hit, {written:,} written)")
                    
                    has_retried_429 = False  # Reset on success
                    self._touch_activity(f"API call #{api_call_count} completed")
                    break  # Success, exit retry loop

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(f"{self.log_prefix}⚡ Interrupted during API call.", force=True)
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"Operation interrupted: waiting for model response ({api_elapsed:.1f}s elapsed)."
                    break

                except Exception as api_error:
                    # Stop spinner before printing error messages
                    if thinking_spinner:
                        thinking_spinner.stop("(╥_╥) error, retrying...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # -----------------------------------------------------------
                    # Surrogate character recovery.  UnicodeEncodeError happens
                    # when the messages contain lone surrogates (U+D800..U+DFFF)
                    # that are invalid UTF-8.  Common source: clipboard paste
                    # from Google Docs or similar rich-text editors.  We sanitize
                    # the entire messages list in-place and retry once.
                    # -----------------------------------------------------------
                    if isinstance(api_error, UnicodeEncodeError) and not getattr(self, '_surrogate_sanitized', False):
                        self._surrogate_sanitized = True
                        if _sanitize_messages_surrogates(messages):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Stripped invalid surrogate characters from messages. Retrying...",
                                force=True,
                            )
                            continue
                        # Surrogates weren't in messages — might be in system
                        # prompt or prefill.  Fall through to normal error path.

                    status_code = getattr(api_error, "status_code", None)
                    error_context = self._extract_api_error_context(api_error)
                    recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                        status_code=status_code,
                        has_retried_429=has_retried_429,
                        error_context=error_context,
                    )
                    if recovered_with_pool:
                        continue
                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(f"{self.log_prefix}🔐 Codex auth refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(f"{self.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                            continue
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, '_anthropic_api_key')
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token
                        if self._try_refresh_anthropic_client_credentials():
                            print(f"{self.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                            continue
                        # Credential refresh didn't help — show diagnostic info
                        key = self._anthropic_api_key
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{self.log_prefix}🔐 Anthropic 401 — authentication failed.")
                        print(f"{self.log_prefix}   Auth method: {auth_method}")
                        print(f"{self.log_prefix}   Token prefix: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (empty or short)")
                        print(f"{self.log_prefix}   Troubleshooting:")
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        print(f"{self.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                        print(f"{self.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                        print(f"{self.log_prefix}     • For API keys: verify at https://console.anthropic.com/settings/keys")
                        print(f"{self.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                        print(f"{self.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_API_KEY \"\"")

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    _error_summary = self._summarize_api_error(api_error)
                    logger.warning(
                        "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        _error_summary,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                    self._vprint(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    self._vprint(f"{self.log_prefix}   📝 Error: {_error_summary}", force=True)
                    if status_code and status_code < 500:
                        _err_body = getattr(api_error, "body", None)
                        _err_body_str = str(_err_body)[:300] if _err_body else None
                        if _err_body_str:
                            self._vprint(f"{self.log_prefix}   📋 Details: {_err_body_str}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")
                    
                    # Check for interrupt before deciding to retry
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"Operation interrupted: handling API error ({error_type}: {self._clean_error_message(str(api_error))}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # Check for 413 payload-too-large BEFORE generic 4xx handler.
                    # A 413 is a payload-size error — the correct response is to
                    # compress history and retry, not abort immediately.
                    status_code = getattr(api_error, "status_code", None)

                    # ── Anthropic Sonnet long-context tier gate ───────────
                    # Anthropic returns HTTP 429 "Extra usage is required for
                    # long context requests" when a Claude Max (or similar)
                    # subscription doesn't include the 1M-context tier.  This
                    # is NOT a transient rate limit — retrying or switching
                    # credentials won't help.  Reduce context to 200k (the
                    # standard tier) and compress.
                    # Only applies to Sonnet — Opus 1M is general access.
                    _is_long_context_tier_error = (
                        status_code == 429
                        and "extra usage" in error_msg
                        and "long context" in error_msg
                        and "sonnet" in self.model.lower()
                    )
                    if _is_long_context_tier_error:
                        _reduced_ctx = 200000
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length
                        if old_ctx > _reduced_ctx:
                            compressor.context_length = _reduced_ctx
                            compressor.threshold_tokens = int(
                                _reduced_ctx * compressor.threshold_percent
                            )
                            compressor._context_probed = True
                            # Don't persist — this is a subscription-tier
                            # limitation, not a model capability.  If the user
                            # later enables extra usage the 1M limit should
                            # come back automatically.
                            compressor._context_probe_persistable = False
                            self._vprint(
                                f"{self.log_prefix}⚠️  Anthropic long-context tier "
                                f"requires extra usage — reducing context: "
                                f"{old_ctx:,} → {_reduced_ctx:,} tokens",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts <= max_compression_attempts:
                            original_len = len(messages)
                            messages, active_system_prompt = self._compress_context(
                                messages, system_message,
                                approx_tokens=approx_tokens,
                                task_id=effective_task_id,
                            )
                            if len(messages) < original_len or old_ctx > _reduced_ctx:
                                self._emit_status(
                                    f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                                    f"(was {old_ctx:,}), retrying..."
                                )
                                time.sleep(2)
                                restart_with_compressed_messages = True
                                break
                        # Fall through to normal error handling if compression
                        # is exhausted or didn't help.

                    # Eager fallback for rate-limit errors (429 or quota exhaustion).
                    # When a fallback model is configured, switch immediately instead
                    # of burning through retries with exponential backoff -- the
                    # primary provider won't recover within the retry window.
                    is_rate_limited = (
                        status_code == 429
                        or "rate limit" in error_msg
                        or "too many requests" in error_msg
                        or "rate_limit" in error_msg
                        or "usage limit" in error_msg
                        or "quota" in error_msg
                    )
                    if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                        # Don't eagerly fallback if credential pool rotation may
                        # still recover.  The pool's retry-then-rotate cycle needs
                        # at least one more attempt to fire — jumping to a fallback
                        # provider here short-circuits it.
                        pool = self._credential_pool
                        pool_may_recover = pool is not None and pool.has_available()
                        if not pool_may_recover:
                            self._emit_status("⚠️ Rate limited — switching to fallback provider...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                continue

                    is_payload_too_large = (
                        status_code == 413
                        or 'request entity too large' in error_msg
                        or 'payload too large' in error_msg
                        or 'error code: 413' in error_msg
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True
                            }
                        self._emit_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}413 payload too large. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "Request payload too large (413). Cannot compress further.",
                                "partial": True
                            }

                    # Check for context-length errors BEFORE generic 4xx handler.
                    # Local backends (LM Studio, Ollama, llama.cpp) often return
                    # HTTP 400 with messages like "Context size has been exceeded"
                    # which must trigger compression, not an immediate abort.
                    is_context_length_error = any(phrase in error_msg for phrase in [
                        'context length', 'context size', 'maximum context',
                        'token limit', 'too many tokens', 'reduce the length',
                        'exceeds the limit', 'context window',
                        'request entity too large',  # OpenRouter/Nous 413 safety net
                        'prompt is too long',  # Anthropic: "prompt is too long: N tokens > M maximum"
                        'prompt exceeds max length',  # Z.AI / GLM: generic 400 overflow wording
                    ])

                    # Fallback heuristic: Anthropic sometimes returns a generic
                    # 400 invalid_request_error with just "Error" as the message
                    # when the context is too large.  If the error message is very
                    # short/generic AND the session is large, treat it as a
                    # probable context-length error and attempt compression rather
                    # than aborting.  This prevents an infinite failure loop where
                    # each failed message gets persisted, making the session even
                    # larger. (#1630)
                    if not is_context_length_error and status_code == 400:
                        ctx_len = getattr(getattr(self, 'context_compressor', None), 'context_length', 200000)
                        is_large_session = approx_tokens > ctx_len * 0.4 or len(api_messages) > 80
                        is_generic_error = len(error_msg.strip()) < 30  # e.g. just "error"
                        if is_large_session and is_generic_error:
                            is_context_length_error = True
                            self._vprint(
                                f"{self.log_prefix}⚠️  Generic 400 with large session "
                                f"(~{approx_tokens:,} tokens, {len(api_messages)} msgs) — "
                                f"treating as probable context overflow.",
                                force=True,
                            )

                    # Server disconnects on large sessions are often caused by
                    # the request exceeding the provider's context/payload limit
                    # without a proper HTTP error response.  Treat these as
                    # context-length errors to trigger compression rather than
                    # burning through retries that will all fail the same way.
                    # This breaks the death spiral: disconnect → no token data
                    # → no compression → bigger session → more disconnects.
                    # (#2153)
                    if not is_context_length_error and not status_code:
                        _is_server_disconnect = (
                            'server disconnected' in error_msg
                            or 'peer closed connection' in error_msg
                            or error_type in ('ReadError', 'RemoteProtocolError', 'ServerDisconnectedError')
                        )
                        if _is_server_disconnect:
                            ctx_len = getattr(getattr(self, 'context_compressor', None), 'context_length', 200000)
                            _is_large = approx_tokens > ctx_len * 0.6 or len(api_messages) > 200
                            if _is_large:
                                is_context_length_error = True
                                self._vprint(
                                    f"{self.log_prefix}⚠️  Server disconnected with large session "
                                    f"(~{approx_tokens:,} tokens, {len(api_messages)} msgs) — "
                                    f"treating as context-length error, attempting compression.",
                                    force=True,
                                )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # Try to parse the actual limit from the error message
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}⚠️  Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})", force=True)
                        else:
                            # Step down to the next probe tier
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.context_length = new_ctx
                            compressor.threshold_tokens = int(new_ctx * compressor.threshold_percent)
                            compressor._context_probed = True
                            # Only persist limits parsed from the provider's
                            # error message (a real number).  Guessed fallback
                            # tiers from get_next_probe_tier() should stay
                            # in-memory only — persisting them pollutes the
                            # cache with wrong values.
                            compressor._context_probe_persistable = bool(
                                parsed_limit and parsed_limit == new_ctx
                            )
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded — stepping down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  Context length exceeded at minimum tier — attempting compression...", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logging.error(f"{self.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                                "partial": True
                            }
                        self._emit_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._emit_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                            time.sleep(2)  # Brief pause between compression retries
                            restart_with_compressed_messages = True
                            break
                        else:
                            # Can't compress further and already at minimum tier
                            self._vprint(f"{self.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                            self._vprint(f"{self.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                            logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                                "partial": True
                            }

                    # Check for non-retryable client errors (4xx HTTP status codes).
                    # These indicate a problem with the request itself (bad model ID,
                    # invalid API key, forbidden, etc.) and will never succeed on retry.
                    # Note: 413 and context-length errors are excluded — handled above.
                    # 429 (rate limit) is transient and MUST be retried with backoff.
                    # 529 (Anthropic overloaded) is also transient.
                    # Also catch local validation errors (ValueError, TypeError) — these
                    # are programming bugs, not transient failures.
                    # Exclude UnicodeEncodeError — it's a ValueError subclass but is
                    # handled separately by the surrogate sanitization path above.
                    _RETRYABLE_STATUS_CODES = {413, 429, 529}
                    is_local_validation_error = (
                        isinstance(api_error, (ValueError, TypeError))
                        and not isinstance(api_error, UnicodeEncodeError)
                    )
                    # Detect generic 400s from Anthropic OAuth (transient server-side failures).
                    # Real invalid_request_error responses include a descriptive message;
                    # transient ones contain only "Error" or are empty. (ref: issue #1608)
                    _err_body = getattr(api_error, "body", None) or {}
                    _err_message = (_err_body.get("error", {}).get("message", "") if isinstance(_err_body, dict) else "")
                    _is_generic_400 = (status_code == 400 and _err_message.strip().lower() in ("error", ""))
                    is_client_status_error = isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in _RETRYABLE_STATUS_CODES and not _is_generic_400
                    is_client_error = (is_local_validation_error or is_client_status_error or any(phrase in error_msg for phrase in [
                        'error code: 401', 'error code: 403',
                        'error code: 404', 'error code: 422',
                        'is not a valid model', 'invalid model', 'model not found',
                        'invalid api key', 'invalid_api_key', 'authentication',
                        'unauthorized', 'forbidden', 'not found',
                    ])) and not is_context_length_error

                    if is_client_error:
                        # Try fallback before aborting — a different provider
                        # may not have the same issue (rate limit, auth, etc.)
                        self._emit_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        self._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                        self._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{self._summarize_api_error(api_error)}"
                        )
                        self._vprint(f"{self.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                        self._vprint(f"{self.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                        self._vprint(f"{self.log_prefix}   🌐 Endpoint: {_base}", force=True)
                        # Actionable guidance for common auth errors
                        if status_code in (401, 403) or "unauthorized" in error_msg or "forbidden" in error_msg or "permission" in error_msg:
                            if _provider == "openai-codex" and status_code == 401:
                                self._vprint(f"{self.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                self._vprint(f"{self.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                self._vprint(f"{self.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                self._vprint(f"{self.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            else:
                                self._vprint(f"{self.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                                self._vprint(f"{self.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                                self._vprint(f"{self.log_prefix}      • Does your account have access to {_model}?", force=True)
                                if "openrouter" in str(_base).lower():
                                    self._vprint(f"{self.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                        logging.error(f"{self.log_prefix}Non-retryable client error: {api_error}")
                        # Skip session persistence when the error is likely
                        # context-overflow related (status 400 + large session).
                        # Persisting the failed user message would make the
                        # session even larger, causing the same failure on the
                        # next attempt. (#1630)
                        if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                            self._vprint(
                                f"{self.log_prefix}⚠️  Skipping session persistence "
                                f"for large failed session to prevent growth loop.",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # Before falling back, try rebuilding the primary
                        # client once for transient transport errors (stale
                        # connection pool, TCP reset).  Only attempted once
                        # per API call block.
                        if not primary_recovery_attempted and self._try_recover_primary_transport(
                            api_error, retry_count=retry_count, max_retries=max_retries,
                        ):
                            primary_recovery_attempted = True
                            retry_count = 0
                            continue
                        # Try fallback before giving up entirely
                        self._emit_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            continue
                        _final_summary = self._summarize_api_error(api_error)
                        if is_rate_limited:
                            self._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                        else:
                            self._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                        self._vprint(f"{self.log_prefix}   💀 Final error: {_final_summary}", force=True)

                        # Detect SSE stream-drop pattern (e.g. "Network
                        # connection lost") and surface actionable guidance.
                        # This typically happens when the model generates a
                        # very large tool call (write_file with huge content)
                        # and the proxy/CDN drops the stream mid-response.
                        _is_stream_drop = (
                            not getattr(api_error, "status_code", None)
                            and any(p in error_msg for p in (
                                "connection lost", "connection reset",
                                "connection closed", "network connection",
                                "network error", "terminated",
                            ))
                        )
                        if _is_stream_drop:
                            self._vprint(
                                f"{self.log_prefix}   💡 The provider's stream "
                                f"connection keeps dropping. This often happens "
                                f"when the model tries to write a very large "
                                f"file in a single tool call.",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      Try asking the model "
                                f"to use execute_code with Python's open() for "
                                f"large files, or to write the file in smaller "
                                f"sections.",
                                force=True,
                            )

                        logging.error(
                            "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                            self.log_prefix, max_retries, _final_summary,
                            _provider, _model, len(api_messages), f"{approx_tokens:,}",
                        )
                        self._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                        self._persist_session(messages, conversation_history)
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                        if _is_stream_drop:
                            _final_response += (
                                "\n\nThe provider's stream connection keeps "
                                "dropping — this often happens when generating "
                                "very large tool call responses (e.g. write_file "
                                "with long content). Try asking me to use "
                                "execute_code with Python's open() for large "
                                "files, or to write in smaller sections."
                            )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _final_summary,
                        }

                    # For rate limits, respect the Retry-After header if present
                    _retry_after = None
                    if is_rate_limited:
                        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                        if _resp_headers and hasattr(_resp_headers, "get"):
                            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                            if _ra_raw:
                                try:
                                    _retry_after = min(int(_ra_raw), 120)  # Cap at 2 minutes
                                except (TypeError, ValueError):
                                    pass
                    wait_time = _retry_after if _retry_after else min(2 ** retry_count, 60)
                    if is_rate_limited:
                        self._emit_status(f"⏱️ Rate limit reached. Waiting {wait_time}s before retry (attempt {retry_count + 1}/{max_retries})...")
                    else:
                        self._emit_status(f"⏳ Retrying in {wait_time}s (attempt {retry_count}/{max_retries})...")
                    logger.warning(
                        "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    # Sleep in small increments so we can respond to interrupts quickly
                    # instead of blocking the entire wait_time in one sleep() call
                    sleep_end = time.time() + wait_time
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # Check interrupt every 200ms
            
            # If the API call was interrupted, skip response processing
            if interrupted:
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # Count compression restarts toward the retry limit to prevent
                # infinite loops when compression reduces messages but not enough
                # to fit the context window.
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                continue

            # Guard: if all retries exhausted without a successful response
            # (e.g. repeated context-length errors that exhausted retry_count),
            # the `response` variable is still None. Break out cleanly.
            if response is None:
                print(f"{self.log_prefix}❌ All API retries exhausted with no successful response.")
                self._persist_session(messages, conversation_history)
                break

            try:
                if self.api_mode == "codex_responses":
                    assistant_message, finish_reason = self._normalize_codex_response(response)
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=self._is_anthropic_oauth
                    )
                else:
                    assistant_message = response.choices[0].message
                
                # Normalize content to string — some OpenAI-compatible servers
                # (llama-server, etc.) return content as a dict or list instead
                # of a plain string, which crashes downstream .strip() calls.
                if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                    elif isinstance(raw, list):
                        # Multimodal content list — extract text parts
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    _assistant_text = assistant_message.content or ""
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        usage=self._usage_summary_for_api_request_hook(response),
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
                except Exception:
                    pass

                # Handle assistant response
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # Notify progress callback of model's thinking (used by subagent
                # delegation to relay the child's reasoning to the parent display).
                if (assistant_message.content and self.tool_progress_callback):
                    _think_text = assistant_message.content.strip()
                    # Strip reasoning XML tags that shouldn't leak to parent display
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    # For subagents: relay first line to parent display (existing behaviour).
                    # For all agents with a structured callback: emit reasoning.available event.
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line and getattr(self, '_delegate_depth', 0) > 0:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                    elif _think_text:
                        try:
                            self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                        except Exception:
                            pass
                
                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    if not hasattr(self, '_incomplete_scratchpad_retries'):
                        self._incomplete_scratchpad_retries = 0
                    self._incomplete_scratchpad_retries += 1
                    
                    self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        # Don't add the broken message, just retry
                        continue
                    else:
                        # Max retries - discard this turn and save as partial
                        self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        
                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                        }
                
                # Reset incomplete scratchpad counter on clean response
                if hasattr(self, '_incomplete_scratchpad_retries'):
                    self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    if not hasattr(self, "_codex_incomplete_retries"):
                        self._codex_incomplete_retries = 0
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

                    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                        last_msg = messages[-1] if messages else None
                        # Duplicate detection: two consecutive incomplete assistant
                        # messages with identical content AND reasoning are collapsed.
                        # For reasoning-only messages (codex_reasoning_items differ but
                        # visible content/reasoning are both empty), we also compare
                        # the encrypted items to avoid silently dropping new state.
                        last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # Validate tool call names - detect model hallucinations
                    # Repair mismatched tool names before validating
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        if not hasattr(self, '_invalid_tool_retries'):
                            self._invalid_tool_retries = 0
                        self._invalid_tool_retries += 1

                        # Return helpful error to model — model can self-correct next turn
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                        self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                        if self._invalid_tool_retries >= 3:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }

                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    # Reset retry counter on successful tool call validation
                    if hasattr(self, '_invalid_tool_retries'):
                        self._invalid_tool_retries = 0
                    
                    # Validate tool call arguments are valid JSON
                    # Handle empty strings as empty objects (common model quirk)
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        # Treat empty/whitespace strings as empty object
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1
                        
                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")
                        
                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # Instead of returning partial, inject tool error results so the model can recover.
                            # Using tool results (not user messages) preserves role alternation.
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                            self._invalid_json_retries = 0  # Reset for next attempt
                            
                            # Append the assistant message with its (broken) tool_calls
                            recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(recovery_assistant)
                            
                            # Respond with tool error results for each tool call
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            continue
                    
                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0

                    # ── Post-call guardrails ──────────────────────────
                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # If this turn has both content AND tool_calls, capture the content
                    # as a fallback final response. Common pattern: model delivers its
                    # answer and calls memory/skill tools as a side-effect in the same
                    # turn. If the follow-up turn after tools is empty, we use this.
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # Only mute subsequent output when EVERY tool call in
                        # this turn is post-response housekeeping (memory, todo,
                        # skill_manage, etc.).  If any substantive tool is present
                        # (search_files, read_file, write_file, terminal, ...),
                        # keep output visible so the user sees progress.
                        _HOUSEKEEPING_TOOLS = frozenset({
                            "memory", "todo", "skill_manage", "session_search",
                        })
                        _all_housekeeping = all(
                            tc.function.name in _HOUSEKEEPING_TOOLS
                            for tc in assistant_message.tool_calls
                        )
                        if _all_housekeeping and self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self.quiet_mode:
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    messages.append(assistant_msg)

                    # Close any open streaming display (response box, reasoning
                    # box) before tool execution begins.  Intermediate turns may
                    # have streamed early content that opened the response box;
                    # flushing here prevents it from wrapping tool feed lines.
                    # Only signal the display callback — TTS (_stream_callback)
                    # should NOT receive None (it uses None as end-of-stream).
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    # Signal that a paragraph break is needed before the next
                    # streamed text.  We don't emit it immediately because
                    # multiple consecutive tool iterations would stack up
                    # redundant blank lines.  Instead, _fire_stream_delta()
                    # will prepend a single "\n\n" the next time real text
                    # arrives.
                    self._stream_needs_break = True

                    # Refund the iteration if the ONLY tool(s) called were
                    # execute_code (programmatic tool calling).  These are
                    # cheap RPC-style calls that shouldn't eat the budget.
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # Use real token counts from the API response to decide
                    # compression.  prompt_tokens + completion_tokens is the
                    # actual context size the provider reported plus the
                    # assistant turn — a tight lower bound for the next prompt.
                    # Tool results appended above aren't counted yet, but the
                    # threshold (default 50%) leaves ample headroom; if tool
                    # results push past it, the next API call will report the
                    # real total and trigger compression then.
                    #
                    # If last_prompt_tokens is 0 (stale after API disconnect
                    # or provider returned no usage data), fall back to rough
                    # estimate to avoid missing compression.  Without this,
                    # a session can grow unbounded after disconnects because
                    # should_compress(0) never fires.  (#2153)
                    _compressor = self.context_compressor
                    if _compressor.last_prompt_tokens > 0:
                        _real_tokens = (
                            _compressor.last_prompt_tokens
                            + _compressor.last_completion_tokens
                        )
                    else:
                        _real_tokens = estimate_messages_tokens_rough(messages)

                    # ── Context pressure warnings (user-facing only) ──────────
                    # Notify the user (NOT the LLM) as context approaches the
                    # compaction threshold.  Thresholds are relative to where
                    # compaction fires, not the raw context window.
                    # Does not inject into messages — just prints to CLI output
                    # and fires status_callback for gateway platforms.
                    if _compressor.threshold_tokens > 0:
                        _compaction_progress = _real_tokens / _compressor.threshold_tokens
                        if _compaction_progress >= 0.85 and not self._context_pressure_warned:
                            self._context_pressure_warned = True
                            self._emit_context_pressure(_compaction_progress, _compressor)

                    if self.compression_enabled and _compressor.should_compress(_real_tokens):
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history so
                        # _flush_messages_to_session_db writes compressed messages
                        # to the new session (see preflight compression comment).
                        conversation_history = None
                    
                    # Save session log incrementally (so progress is visible even if interrupted)
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # If the previous turn already delivered real content alongside
                        # tool calls (e.g. "You're welcome!" + memory save), the model
                        # has nothing more to say. Use the earlier content immediately
                        # instead of wasting API calls on retries that won't help.
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback:
                            logger.debug("Empty follow-up after tool calls — using prior turn content as final response")
                            self._last_content_with_tools = None
                            self._empty_content_retries = 0
                            for i in range(len(messages) - 1, -1, -1):
                                msg = messages[i]
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    tool_names = []
                                    for tc in msg["tool_calls"]:
                                        if not tc or not isinstance(tc, dict): continue
                                        fn = tc.get("function", {})
                                        tool_names.append(fn.get("name", "unknown"))
                                    msg["content"] = f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                    break
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # Reasoning-only response: the model produced thinking
                        # but no visible content.  This is a valid response —
                        # keep reasoning in its own field and set content to
                        # "(empty)" so every provider accepts the message.
                        # No retries needed.
                        reasoning_text = self._extract_reasoning(assistant_message)
                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        assistant_msg["content"] = "(empty)"
                        messages.append(assistant_msg)

                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            self._vprint(f"{self.log_prefix}ℹ️  Reasoning-only response (no visible content). Reasoning: {reasoning_preview}")
                        else:
                            self._vprint(f"{self.log_prefix}ℹ️  Empty response (no content or reasoning).")

                        final_response = "(empty)"
                        break
                    
                    # Reset retry counter/signature on successful content
                    if hasattr(self, '_empty_content_retries'):
                        self._empty_content_retries = 0
                    self._last_empty_content_signature = None

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                        messages.append(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    if truncated_response_prefix:
                        final_response = truncated_response_prefix + final_response
                        truncated_response_prefix = ""
                        length_continue_retries = 0
                    
                    # Strip <think> blocks from user-facing response (keep raw in messages for trajectory)
                    final_response = self._strip_think_blocks(final_response).strip()
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    messages.append(final_msg)
                    
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                if self.verbose_logging:
                    logging.exception("Detailed error information:")
                
                # If an assistant message with tool_calls was already appended,
                # the API expects a role="tool" result for every tool_call_id.
                # Fill in error results for any that weren't answered yet.
                pending_handled = False
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                messages.append(err_msg)
                        pending_handled = True
                    break
                
                # Non-tool errors don't need a synthetic message injected.
                # The error is already printed to the user (line above), and
                # the retry loop continues.  Injecting a fake user/assistant
                # message pollutes history, burns tokens, and risks violating
                # role-alternation invariants.

                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    # Append as assistant so the history stays valid for
                    # session resume (avoids consecutive user messages).
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ):
            if self.iteration_budget.remaining <= 0 and not self.quiet_mode:
                print(f"\n⚠️  Iteration budget exhausted ({self.iteration_budget.used}/{self.iteration_budget.max_total} iterations used)")
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations

        # Save trajectory if enabled
        self._save_trajectory(messages, user_message, completed)

        # Clean up VM and browser for this task after conversation completes
        self._cleanup_task_resources(effective_task_id)

        # Persist session to both JSON log and SQLite
        self._persist_session(messages, conversation_history)


        # Plugin hook: post_llm_call
        # Fired once per turn after the tool-calling loop completes.
        # Plugins can use this to persist conversation data (e.g. sync
        # to an external memory system).
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        # Extract reasoning from the last assistant message (if any)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # Build result with interrupt info if applicable
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # True only when stopped due to invalid tool calls
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        self._response_was_previewed = False
        
        # Include interrupt message if one triggered the interrupt
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # Clear interrupt state after handling
        self.clear_interrupt()

        # Clear stream callback so it doesn't leak into future calls
        self._stream_callback = None

        # Check skill trigger NOW — based on how many tool iterations THIS turn used.
        _should_review_skills = False
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            self._iters_since_skill = 0

        # External memory provider: sync the completed turn + queue next prefetch.
        # Use original_user_message (clean input) — user_message may contain
        # injected skill content that bloats / breaks provider queries.
        if self._memory_manager and final_response and original_user_message:
            try:
                self._memory_manager.sync_all(original_user_message, final_response)
                self._memory_manager.queue_prefetch_all(original_user_message)
            except Exception:
                pass

        # Background memory/skill review — runs AFTER the response is delivered
        # so it never competes with the user's task for model attention.
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # Background review is best-effort

        # Note: Memory provider on_session_end() + shutdown_all() are NOT
        # called here — run_conversation() is called once per user message in
        # multi-turn sessions. Shutting down after every turn would kill the
        # provider before the second message. Actual session-end cleanup is
        # handled by the CLI (atexit / /reset) and gateway (session expiry /
        # _reset_session).

        # Plugin hook: on_session_end
        # Fired at the very end of every run_conversation call.
        # Plugins can use this for cleanup, flushing buffers, etc.
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=completed,
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """
        Simple chat interface that returns just the final response.

        Args:
            message (str): User message
            stream_callback: Optional callback invoked with each text delta during streaming.

        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-sonnet-4.6.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection. Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses. Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    # Handle tool listing
    if list_tools:
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 50)
        
        # Show new toolsets system
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        # Group by category
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        # Print basic toolsets
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")
        
        # Print composite toolsets
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")
        
        # Print scenario-specific toolsets
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")
        
        
        # Show legacy toolset compatibility
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # Parse toolset selection arguments
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")
    
    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Use provided query or default to Python 3.13 example
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    # Run conversation
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    # Save sample trajectory to UUID-named file if requested
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"
        
        # Convert messages to trajectory format (same as batch_runner)
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # Pretty-print JSON with indent for readability
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    fire.Fire(main)
