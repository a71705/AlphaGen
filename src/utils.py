# -*- coding: utf-8 -*-
import logging
from typing import List, Optional as TypingOptional, Any # Use TypingOptional to avoid conflict

# pyparsing imports will be below Node class as per instructions
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
logger_pyparsing = logging.getLogger(__name__ + ".pyparsing") # Dedicated logger for parsing setup

try:
    from pyparsing import (
        Word, alphas, alphanums, nums, Suppress, Group, delimitedList,
        Forward, Optional as PyparsingOptional, Keyword, ParserElement, Combine, Literal,
        ParseException, ParseResults, pyparsing_common
    )
    ParserElement.enablePackrat()
    logger_pyparsing.info("Pyparsing Packrat优化已启用。")

    # 定义基础语法元素
    identifier = Word(alphas + "_", alphanums + "_").setName("identifier")

    # 使用 pyparsing_common.number, 它能处理整数、实数以及可选的正负号
    # 它默认会将匹配到的字符串转换为 float 或 int
    number = pyparsing_common.number().setName("number")

    # Forward声明表达式，用于递归定义
    expr = Forward().setName("expression")

    # 定义函数调用结构
    lparen = Suppress("(")
    rparen = Suppress(")")

    # arg可以是expr本身（递归），或一个identifier，或一个number
    # 注意：这里的顺序很重要，expr 应该优先于 identifier 和 number，以正确处理嵌套表达式
    # 然而，由于 expr 最终会包含 identifier 和 number，pyparsing的默认行为是尝试最长的匹配。
    # 为了让解析动作更容易处理，我们将对 identifier 和 number 单独设置动作，
    # function_call 的动作将期望其参数已经是 Node 对象。

    # 为基本元素定义创建Node的动作
    def identifier_to_node(s,l,t):
        logger_pyparsing.debug(f"Identifier action: t={t}")
        return Node(t[0])

    def number_to_node(s,l,t):
        logger_pyparsing.debug(f"Number action: t={t}")
        # pyparsing_common.number already converts to float/int
        return Node(t[0])

    identifier.setParseAction(identifier_to_node)
    number.setParseAction(number_to_node)

    # 现在定义 arg，它将使用已经带有Node转换动作的 expr, identifier, number
    arg = expr # arg 直接就是 expr，因为 expr 会包含 identifier 和 number

    # Group确保函数名和参数列表被视为一个整体，并为它们设置一个特定的解析动作
    # 函数名 (identifier) 本身不应触发其全局的 identifier_to_node 动作，
    # 而应作为字符串被 function_call_to_node 处理。
    # 为此，我们可以用 ~expr + identifier 来确保函数名被当做纯粹的标识符，
    # 或者在 function_call_to_node 中处理 tokens[0] 可能已经是Node的情况。
    # 更简单的方法是让function_call中的identifier不带parseAction，或者复制一个不带action的identifier。
    func_name_identifier = Word(alphas + "_", alphanums + "_").setName("func_name_identifier")

    function_call = Group(
        func_name_identifier + lparen + PyparsingOptional(delimitedList(arg)) + rparen
    ).setName("function_call")

    def function_call_to_node(s,l,t):
        logger_pyparsing.debug(f"Function call action: t={t}")
        # t is a ParseResults object from the Group
        # t[0] is the result of the Group: [func_name_str, arg1_node, arg2_node, ...]
        grouped_tokens = t[0]
        func_name = grouped_tokens[0] # This is a string from func_name_identifier
        children_nodes = list(grouped_tokens[1:]) # These should already be Node objects
        logger_pyparsing.debug(f"Creating function Node: {func_name}, children: {children_nodes}")
        return Node(func_name, children_nodes)

    function_call.setParseAction(function_call_to_node)

    # 表达式的核心可以是函数调用、单个标识符或单个数字
    # 顺序很重要：尝试匹配最复杂的结构（function_call）优先
    expr <<= (function_call | identifier | number)

    # 移除 simple_parser，现在 expr 是主解析器
    # simple_parser = identifier | number # No longer needed

except ImportError:
    logging.getLogger(__name__).critical("Pyparsing模块未安装，解析功能将不可用。请安装pyparsing: pip install pyparsing")
    ParserElement = object
    ParseException = Exception # Placeholder for ParseException
    identifier = None
    number = None
    expr = None
except Exception as e: # Catch any other pyparsing setup errors
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

    if expr is None: # Check if pyparsing failed to import/initialize
        logger.error("由于pyparsing未成功加载或初始化，无法解析表达式。")
        return None

    try:
        # 使用完整的 expr 解析器
        # parseAll=True 确保整个字符串都被解析
        parsed_result = expr.parseString(expression_str, parseAll=True)

        # 经过setParseAction后，parsed_result[0] 应该直接是根Node对象
        if parsed_result and isinstance(parsed_result[0], Node):
             logger.info(f"表达式 '{expression_str}' 成功解析为AST: {parsed_result[0]}")
             return parsed_result[0]
        else:
            # 这种情况理论上不应发生，如果解析成功且动作正确，结果应为Node
            logger.error(f"表达式 '{expression_str}' 解析后未得到预期的Node。Result: {parsed_result}")
            return None

    except ParseException as pe:
        # ParseException 提供了更详细的错误信息，如位置
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

    # 叶子节点 (无子节点): 直接返回其值的字符串形式
    if not node.children:
        expr_str = str(node.value)
        logger.debug(f"从叶子节点 {node} 生成表达式: '{expr_str}'")
        return expr_str
    else:
        # 内部节点 (有子节点，代表函数调用)
        child_expressions = [generate_expression_from_ast(child) for child in node.children]
        # 处理函数调用没有参数的情况
        if not child_expressions: # No children were passed, or all children resulted in empty strings (less likely here)
            child_expressions_str = ""
        else:
            child_expressions_str = ", ".join(child_expressions)

        # node.value 应该是函数名 (字符串)
        expr_str = f"{str(node.value)}({child_expressions_str})"
        logger.debug(f"从内部节点 {node} 生成表达式: '{expr_str}'")
        return expr_str

# Example usage and tests (can be placed in if __name__ == '__main__':)
if __name__ == '__main__':
    # 配置顶层日志记录器，以便utils.py中的日志可以输出
    # 如果调用此脚本的外部模块已经配置了logging，则这里可能不需要再次配置
    # 但为了独立测试 utils.py，进行基本配置是好的
    if not logging.getLogger().hasHandlers(): # Avoid adding multiple handlers if already configured
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_logger = logging.getLogger(__name__) # Logger for this __main__ block

    # Test Node class
    node_close = Node("close")
    node_10 = Node(10)
    node_rank = Node("rank", [node_close, node_10])
    main_logger.info(f"Node close: {node_close}")
    main_logger.info(f"Node 10: {node_10}")
    main_logger.info(f"Node rank: {node_rank}")

    # Test parsing
    test_expressions = [
        "close",
        "123",
        "-5.5",
        "ts_rank(close, 10)",
        "add(open, close)",
        "complex_func(ts_rank(close, 10), add(open, -5), 3.14)",
        "func_no_args()",
        "func_one_arg(close)",
        "a", # simple identifier
        "1", # simple number
        "1.0",
        "-1",
        "-1.0",
        "log(close)",
        "sma(high, 10)",
        "   spaced_func  (  arg1 ,  arg2   )  " # Test with spaces
    ]

    for expr_str in test_expressions:
        main_logger.info(f"\n--- Testing Expression: '{expr_str}' ---")
        ast_root = parse_expression_to_ast(expr_str)
        if ast_root:
            main_logger.info(f"Successfully parsed to AST: {ast_root}")
            generated_expr = generate_expression_from_ast(ast_root)
            main_logger.info(f"Generated expression from AST: '{generated_expr}'")
            # Basic check: re-parse the generated expression
            # Note: This re-parse check might fail if generate_expression_from_ast doesn't perfectly normalize (e.g. spacing)
            # For a more robust check, you'd compare ASTs or use a canonical string representation.
            # ast_reparsed = parse_expression_to_ast(generated_expr)
            # if ast_reparsed == ast_root:
            #     main_logger.info(f"Re-parsing and AST comparison successful for: '{generated_expr}'")
            # else:
            #     main_logger.error(f"AST Mismatch after re-parsing! Original AST: {ast_root}, Reparsed AST: {ast_reparsed}")
        else:
            main_logger.error(f"Failed to parse expression: '{expr_str}'")

    # Test invalid expressions
    invalid_expressions = [
        "func(", # Missing closing parenthesis
        "func(a,)", # Trailing comma
        "(a+b)", # Operators not yet supported
        "1.2.3" # Invalid number format
    ]
    for expr_str in invalid_expressions:
        main_logger.info(f"\n--- Testing Invalid Expression: '{expr_str}' ---")
        ast_root = parse_expression_to_ast(expr_str)
        if ast_root:
            main_logger.error(f"Incorrectly parsed invalid expression '{expr_str}' to AST: {ast_root}")
        else:
            main_logger.info(f"Correctly failed to parse invalid expression: '{expr_str}'")

    # Test generate_expression_from_ast with None
    main_logger.info(f"\n--- Testing generate_expression_from_ast(None) ---")
    main_logger.info(f"Generated from None: '{generate_expression_from_ast(None)}'")
```
