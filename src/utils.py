# -*- coding: utf-8 -*-
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional as TypingOptional, Any, TYPE_CHECKING

# pyparsing imports and re will be below setup_logging or where they are currently,
# as per standard practice of grouping imports.
# TYPE_CHECKING is used by other parts of the file, so it's fine here.

# --- Logging Setup Function ---
def setup_logging(base_log_dir: str = "logs",
                  log_level: int = logging.INFO,
                  logger_name: TypingOptional[str] = None) -> logging.Logger:
    """
    配置系统日志，按日期创建文件夹。
    可以为特定的logger（如果提供了logger_name）或根logger配置处理器。

    参数:
        base_log_dir (str): 基础日志目录的路径。
        log_level (int): 要设置的日志级别 (例如 logging.INFO, logging.DEBUG)。
        logger_name (Optional[str]): 如果提供，则配置此名称的logger；否则配置根logger。

    返回:
        logging.Logger: 配置好的logger实例。
    """
    # 此处的 logger_setup_internal 是为了能在早期捕获日志目录创建问题时使用
    logger_setup_internal = logging.getLogger(__name__ + ".setup_logging_internal")

    log_date_dir = os.path.join(base_log_dir, datetime.now().strftime("%Y-%m-%d"))
    try:
        os.makedirs(log_date_dir, exist_ok=True)
    except OSError as e:
        # 如果在尝试创建日志目录时发生错误 (例如权限问题)，
        # 打印到stderr并尝试继续，但日志可能无法写入文件。
        # 使用 print 是因为此时自定义的 logger 可能还无法写入文件。
        print(f"严重错误：无法创建日志目录 {log_date_dir}: {e}", file=sys.stderr)
        # 对于本系统，我们尝试让控制台日志仍然工作，所以不在此处抛出致命异常。
        # 但这是一个严重问题，应该被注意到。

    system_log_path = os.path.join(log_date_dir, "system.log")
    errors_log_path = os.path.join(log_date_dir, "errors.log")

    # 获取要配置的logger
    target_logger: logging.Logger
    if logger_name:
        target_logger = logging.getLogger(logger_name)
    else:
        target_logger = logging.getLogger() # 获取根logger

    target_logger.setLevel(log_level) # 为选定的logger设置级别

    # 避免重复添加handler (重要，防止多次调用setup_logging时重复输出)
    # 只移除属于当前配置目标logger的FileHandler和StreamHandler
    for handler in list(target_logger.handlers): # Iterate over a copy for safe removal
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            # 为了更精确，可以检查handler是否是之前由此函数添加的，
            # 但简单移除所有已存在的同类型handler通常是可接受的起点。
            target_logger.removeHandler(handler)
            try:
                handler.close() # 关闭处理器以释放文件锁等资源
            except Exception as e_close:
                # 记录关闭handler时可能发生的错误，但不应中断日志设置
                logger_setup_internal.debug(f"关闭旧handler时发生非致命错误: {e_close}")

    # 创建文件格式化器
    # 包含更详细的模块和函数名信息: %(module)s:%(funcName)s:%(lineno)d
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler for all INFO (or configured log_level) and above
    try:
        file_handler = logging.FileHandler(system_log_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        target_logger.addHandler(file_handler)
    except IOError as e_io_sys:
        print(f"警告：无法打开或写入系统日志文件 {system_log_path}: {e_io_sys}", file=sys.stderr)

    # File Handler for ERROR only
    try:
        error_file_handler = logging.FileHandler(errors_log_path, encoding='utf-8')
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(file_formatter) # 使用与主文件日志相同的格式
        target_logger.addHandler(error_file_handler)
    except IOError as e_io_err:
        print(f"警告：无法打开或写入错误日志文件 {errors_log_path}: {e_io_err}", file=sys.stderr)

    # Console Handler (输出到 stdout)
    # 只有在没有其他StreamHandler配置到stdout时才添加，或者如果明确要替换
    # 检查是否已存在输出到 sys.stdout 的 StreamHandler
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in target_logger.handlers
    )
    if not has_stdout_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', # 控制台格式可以简洁一些
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        target_logger.addHandler(console_handler)

    # 针对根logger的首次配置，打印一条概要信息
    if not logger_name and not getattr(target_logger, '_initial_setup_complete', False):
        target_logger.info(f"日志系统已配置。日志级别: {logging.getLevelName(target_logger.level)}。文件日志输出到: {log_date_dir}")
        setattr(target_logger, '_initial_setup_complete', True) # 标记已完成初始设置
    elif logger_name:
        target_logger.info(f"Logger '{logger_name}' 已配置。日志级别: {logging.getLevelName(target_logger.level)}。")

    return target_logger

# --- pyparsing and re imports ---
# These were below Node class in the previous version, keeping them after setup_logging now
# but before other class/function definitions seems reasonable.
import re

# --- Node Class Definition ---
class Node:
    """表示抽象语法树（AST）中的一个节点。"""
    def __init__(self, value: Any, children: TypingOptional[List['Node']] = None): # value can be str, int, float
        """
        初始化AST节点。

        参数:
            value: 节点的值（例如：操作符、函数名、数据字段、常量）。
            children (list['Node'] | None): 一个包含子节点的列表，默认为None，将初始化为空列表。
        """
        self.value = value
        self.children: List['Node'] = children if children is not None else []
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"创建Node: value='{str(self.value)}' (type: {type(self.value).__name__}), children_count={len(self.children)}")

    def __repr__(self) -> str:
        """返回节点的字符串表示形式，方便调试。"""
        if self.children:
            return f"Node(value='{str(self.value)}', children={self.children})"
        return f"Node(value='{str(self.value)}')"

    def __eq__(self, other) -> bool:
        """比较两个Node对象是否相等，主要用于测试。"""
        if not isinstance(other, Node):
            return NotImplemented
        return self.value == other.value and self.children == other.children

# --- Pyparsing Initialization and Base Elements ---
logger_pyparsing = logging.getLogger(__name__ + ".pyparsing")

try:
    from pyparsing import (
        Word, alphas, alphanums, nums, Suppress, Group, delimitedList,
        Forward, Optional as PyparsingOptional, Keyword, ParserElement, Combine, Literal,
        ParseException, ParseResults, pyparsing_common
    )
    ParserElement.enablePackrat()
    logger_pyparsing.info("Pyparsing Packrat优化已启用。")

    identifier = Word(alphas + "_", alphanums + "_").setName("identifier")
    number = pyparsing_common.number().setName("number")
    expr = Forward().setName("expression")
    lparen = Suppress("(")
    rparen = Suppress(")")

    def identifier_to_node(s,l,t):
        logger_pyparsing.debug(f"Identifier action: t={t}")
        return Node(t[0])

    def number_to_node(s,l,t):
        logger_pyparsing.debug(f"Number action: t={t}")
        return Node(t[0])

    identifier.setParseAction(identifier_to_node)
    number.setParseAction(number_to_node)

    arg = expr
    func_name_identifier = Word(alphas + "_", alphanums + "_").setName("func_name_identifier")
    function_call = Group(
        func_name_identifier + lparen + PyparsingOptional(delimitedList(arg)) + rparen
    ).setName("function_call")

    def function_call_to_node(s,l,t):
        logger_pyparsing.debug(f"Function call action: t={t}")
        grouped_tokens = t[0]
        func_name = grouped_tokens[0]
        children_nodes = list(grouped_tokens[1:])
        logger_pyparsing.debug(f"Creating function Node: {func_name}, children: {children_nodes}")
        return Node(func_name, children_nodes)

    function_call.setParseAction(function_call_to_node)
    expr <<= (function_call | identifier | number)

except ImportError:
    logging.getLogger(__name__).critical("Pyparsing模块未安装，解析功能将不可用。请安装pyparsing: pip install pyparsing")
    ParserElement = object
    ParseException = Exception
    identifier = None
    number = None
    expr = None
except Exception as e:
    logging.getLogger(__name__).critical(f"Pyparsing 初始化过程中发生未知错误: {e}", exc_info=True)
    ParserElement = object
    ParseException = Exception
    identifier = None
    number = None
    expr = None

# --- AST Parsing Function (Updated) ---
def parse_expression_to_ast(expression_str: str) -> TypingOptional[Node]:
    """
    将Alpha表达式字符串解析为AST。
    支持函数调用、标识符和数字。

    参数:
        expression_str (str): 要解析的Alpha表达式字符串。

    返回:
        Node | None: 解析成功则返回AST的根节点，否则返回None。
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始解析表达式: '{expression_str}'")

    if expr is None:
        logger.error("由于pyparsing未成功加载或初始化，无法解析表达式。")
        return None

    try:
        parsed_result = expr.parseString(expression_str, parseAll=True)
        if parsed_result and isinstance(parsed_result[0], Node):
             logger.info(f"表达式 '{expression_str}' 成功解析为AST: {parsed_result[0]}")
             return parsed_result[0]
        else:
            logger.error(f"表达式 '{expression_str}' 解析后未得到预期的Node。Result: {parsed_result}")
            return None
    except ParseException as pe:
        logger.error(f"解析表达式 '{expression_str}' 失败 (pyparsing.ParseException): {pe.explain()}")
        logger.debug(f"ParseException details: line:{pe.lineno}, col:{pe.col}, msg:{pe.msg}, offendingToken:{pe.parserElement}")
        return None
    except Exception as e:
        logger.error(f"解析表达式 '{expression_str}' 时发生未知错误: {e}", exc_info=True)
        return None

# --- AST Generation Function (Updated) ---
def generate_expression_from_ast(node: TypingOptional[Node]) -> str:
    """
    将AST结构转换回Alpha表达式字符串。
    支持叶子节点 (标识符/数字) 和代表函数调用的内部节点。

    参数:
        node (Node | None): AST的节点。

    返回:
        str: 从AST生成的表达式字符串。如果输入为None，返回空字符串。
    """
    logger = logging.getLogger(__name__)
    if node is None:
        logger.debug("generate_expression_from_ast 接收到 None 输入，返回空字符串。")
        return ""
    if not isinstance(node, Node):
        logger.warning(f"generate_expression_from_ast接收到的输入不是Node类型: {type(node)}，返回空字符串。")
        return ""

    logger.debug(f"开始从Node生成表达式: {node}")

    if not node.children:
        expr_str = str(node.value)
        logger.debug(f"从叶子节点 {node} 生成表达式: '{expr_str}'")
        return expr_str
    else:
        child_expressions = [generate_expression_from_ast(child) for child in node.children]
        if not child_expressions:
            child_expressions_str = ""
        else:
            child_expressions_str = ", ".join(child_expressions)
        expr_str = f"{str(node.value)}({child_expressions_str})"
        logger.debug(f"从内部节点 {node} 生成表达式: '{expr_str}'")
        return expr_str

# Example usage and tests
if __name__ == '__main__':
    # 基本日志配置，如果 setup_logging 未被外部调用，则使用此配置
    # setup_logging() # 可以取消注释以在此处测试 setup_logging 本身
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 测试 setup_logging
    test_setup_logger = logging.getLogger("TestSetupLogging")
    setup_logging(log_level=logging.DEBUG, logger_name="TestSetupLogging") # 配置特定logger
    test_setup_logger.debug("这是来自 TestSetupLogging logger 的 DEBUG 消息。")
    test_setup_logger.info("这是来自 TestSetupLogging logger 的 INFO 消息。")

    # 配置根 logger (如果需要全局测试)
    # root_logger_configured = setup_logging(log_level=logging.DEBUG)
    # logging.debug("这是来自根 logger 的 DEBUG 消息 (如果已配置)。")


    main_logger = logging.getLogger(__name__)
    main_logger.info("\n--- 开始 utils.py 的 __main__ 测试 ---")

    # Test Node class
    node_close = Node("close")
    node_10 = Node(10)
    node_rank = Node("rank", [node_close, node_10])
    main_logger.info(f"Node close: {node_close}")
    # ... (其余测试代码保持不变，为简洁省略) ...

    test_expressions = [
        "close", "123", "-5.5", "ts_rank(close, 10)", "add(open, close)",
        "complex_func(ts_rank(close, 10), add(open, -5), 3.14)",
        "func_no_args()", "func_one_arg(close)", "a", "1", "1.0", "-1", "-1.0",
        "log(close)", "sma(high, 10)", "   spaced_func  (  arg1 ,  arg2   )  "
    ]
    for expr_str in test_expressions:
        main_logger.info(f"\n--- Testing Expression: '{expr_str}' ---")
        ast_root = parse_expression_to_ast(expr_str)
        if ast_root:
            main_logger.info(f"Successfully parsed to AST: {ast_root}")
            generated_expr = generate_expression_from_ast(ast_root)
            main_logger.info(f"Generated expression from AST: '{generated_expr}'")
        else:
            main_logger.error(f"Failed to parse expression: '{expr_str}'")

    invalid_expressions = [
        "func(", "func(a,)", "(a+b)", "1.2.3"
    ]
    for expr_str in invalid_expressions:
        main_logger.info(f"\n--- Testing Invalid Expression: '{expr_str}' ---")
        ast_root = parse_expression_to_ast(expr_str)
        if ast_root:
            main_logger.error(f"Incorrectly parsed invalid expression '{expr_str}' to AST: {ast_root}")
        else:
            main_logger.info(f"Correctly failed to parse invalid expression: '{expr_str}'")

    main_logger.info(f"\n--- Testing generate_expression_from_ast(None) ---")
    main_logger.info(f"Generated from None: '{generate_expression_from_ast(None)}'")
    main_logger.info("\n--- utils.py 的 __main__ 测试结束 ---")

