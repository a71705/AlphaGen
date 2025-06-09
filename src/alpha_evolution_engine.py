# -*- coding: utf-8 -*-
import logging
import random # 用于后续的Alpha生成和进化操作
import re # 用于模板占位符替换
from typing import Dict, List, Optional, Any # MyPy需要明确的类型提示

# --- 依赖导入 ---
# 尝试相对导入 (适用于作为包一部分导入的场景)
# 如果直接运行此文件或相关模块在PYTHONPATH中，则会尝试绝对导入
try:
    from .utils import Node, generate_expression_from_ast, parse_expression_to_ast # 添加 parse_expression_to_ast
    from .config_manager import ConfigManager
    from .wqb_api_client import WQB_API_Client
    from .result_handler import ResultHandler
except ImportError:
    # Fallback for direct execution or different project structure
    try:
        from utils import Node, generate_expression_from_ast, parse_expression_to_ast # 添加 parse_expression_to_ast
        from config_manager import ConfigManager
        from wqb_api_client import WQB_API_Client
        from result_handler import ResultHandler
    except ImportError as e_abs:
        logging.getLogger(__name__).critical(
            f"无法导入必要的模块 (Node, generate_expression_from_ast, parse_expression_to_ast, ConfigManager, WQB_API_Client, ResultHandler): {e_abs}. "
            "确保这些模块在正确的路径下并且没有循环导入问题。"
        )
        # 定义临时的占位符类/函数
        class Node: # type: ignore
            def __init__(self, value, children=None): self.value = value; self.children = children if children is not None else []
            def __repr__(self): return f"PlaceholderNode({self.value})"
        def generate_expression_from_ast(node: Optional[Node]) -> str: # type: ignore
            if node: return f"ErrorGeneratingExpression({node.value})"
            return "[Error: Node or generate_expression_from_ast not loaded]"
        def parse_expression_to_ast(expr_str: str) -> Optional[Node]: # type: ignore
            logging.getLogger(__name__).error("parse_expression_to_ast 存根被调用，utils 未正确导入。")
            return None
        class ConfigManager: pass # type: ignore
        class WQB_API_Client: pass # type: ignore
        class ResultHandler: pass # type: ignore

# --- Individual 类定义 ---
class Individual:
    """
    表示遗传算法（GA）中的一个Alpha个体（一个潜在的投资策略）。
    """
    def __init__(self,
                 expression_ast: Node,
                 simulation_params: Dict[str, Any],
                 fitness_score: float = 0.0,
                 alpha_id: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        if not isinstance(expression_ast, Node):
            err_msg = f"expression_ast 参数必须是 Node 类型，但收到了 {type(expression_ast)}"
            self.logger.error(err_msg)
            raise TypeError(err_msg)
        if not isinstance(simulation_params, dict):
            err_msg = f"simulation_params 参数必须是字典类型，但收到了 {type(simulation_params)}"
            self.logger.error(err_msg)
            raise TypeError(err_msg)
        self.expression_ast: Node = expression_ast
        self.simulation_params: Dict[str, Any] = simulation_params
        self.fitness_score: float = fitness_score
        self.alpha_id: Optional[str] = alpha_id
        ast_root_value_str = str(expression_ast.value) if expression_ast else "[AST为None]"
        self.logger.debug(
            f"创建Individual: AST根节点值='{ast_root_value_str}', "
            f"模拟参数={simulation_params}, Fitness={fitness_score}, AlphaID='{alpha_id}'"
        )

    def __repr__(self) -> str:
        ast_root_value_str = str(self.expression_ast.value) if self.expression_ast else "[AST为None]"
        ast_repr = f"AST(root_value='{ast_root_value_str}')"
        return (
            f"Individual(alpha_id='{self.alpha_id}', fitness={self.fitness_score:.4f}, "
            f"params={self.simulation_params}, ast_repr={ast_repr})"
        )

    def get_expression_str(self) -> str:
        try:
            if not isinstance(self.expression_ast, Node):
                self.logger.warning("尝试从非Node类型的expression_ast生成表达式字符串。")
                return "[表达式AST无效]"
            return generate_expression_from_ast(self.expression_ast)
        except Exception as e:
            self.logger.error(f"从AST生成表达式字符串时发生错误: {e}", exc_info=True)
            return f"[表达式生成错误: {e}]"

# --- AlphaEvolutionEngine 类定义 ---
class AlphaEvolutionEngine:
    """
    Alpha因子进化引擎。
    （类文档字符串与之前相同）
    """
    def __init__(self,
                 config_manager: ConfigManager,
                 wqb_api_client: WQB_API_Client,
                 result_handler: ResultHandler):
        self.logger = logging.getLogger(__name__)
        self.logger.info("AlphaEvolutionEngine 初始化开始...")

        self.config_manager: ConfigManager = config_manager
        self.wqb_api_client: WQB_API_Client = wqb_api_client
        self.result_handler: ResultHandler = result_handler

        # GA核心参数加载 (与之前相同)
        self.population_size: int = self.config_manager.get("ga.population_size", 100)
        self.max_generations: int = self.config_manager.get("ga.max_generations", 50)
        self.injection_rate: float = self.config_manager.get("ga.injection_rate", 0.1)
        self.min_population_size: int = self.config_manager.get("ga.min_population_size", 20)
        self.max_population_size: int = self.config_manager.get("ga.max_population_size", 200)
        self.target_successful_alphas: int = self.config_manager.get("ga.target_successful_alphas", 50)
        self.alpha_generation_mode: str = self.config_manager.get("ga.alpha_generation_mode", "template_based")

        # 操作符集和数据字段加载 (与之前相同)
        self.terminal_values: List[str] = self.config_manager.get("operator_sets.terminal_values", [])
        self.ts_ops: List[str] = self.config_manager.get("operator_sets.ts_ops", [])
        self.binary_ops: List[str] = self.config_manager.get("operator_sets.binary_ops", [])
        self.unary_ops: List[str] = self.config_manager.get("operator_sets.unary_ops", [])
        self.ts_ops_values: List[int] = self.config_manager.get("operator_sets.ts_ops_values", [5, 10, 20, 60])
        self.group_ops: List[str] = self.config_manager.get("operator_sets.group_ops", [])

        # 模拟参数可选范围加载 (与之前相同)
        self.decay_choices: List[int] = self.config_manager.get("operator_sets.decay_choices", [0, 1, 5, 10])
        self.delay_choices: List[int] = self.config_manager.get("operator_sets.delay_choices", [0, 1, 2])
        self.truncation_choices: List[float] = self.config_manager.get("operator_sets.truncation_choices", [0.01, 0.05])
        self.universe_choices: List[str] = self.config_manager.get("operator_sets.universe_choices", ["TOP1000", "TOP3000"])
        self.neutralization_choices: List[Optional[str]] = self.config_manager.get("operator_sets.neutralization_choices", [None, "market", "industry"])

        self.ga_genes: List[str] = self.config_manager.get("ga.genes", ['decay', 'universe', 'neutralization'])

        # GA状态变量初始化 (与之前相同)
        self.current_population: List[Individual] = []
        self.current_generation: int = 0
        self.total_successful_alphas_found: int = 0

        # 预处理黑名单名称，提高查找效率
        blacklist_dicts = self.config_manager.get("blacklist", [])
        self._processed_blacklist_names: set[str] = {item['name'] for item in blacklist_dicts if isinstance(item, dict) and 'name' in item}
        self.logger.debug(f"预处理后的黑名单名称集合: {self._processed_blacklist_names}")

        self.logger.info(
            f"AlphaEvolutionEngine 初始化完毕。Population Size: {self.population_size}, Max Generations: {self.max_generations}"
        )
        self.logger.debug(f"GA Genes (evolvable simulation params): {self.ga_genes}")
        self.logger.debug(f"Operator sets loaded: ts_ops_count={len(self.ts_ops)}, binary_ops_count={len(self.binary_ops)}")

    def _generate_random_simulation_params(self) -> Dict[str, Any]:
        """
        随机生成一套用于Alpha模拟的参数。

        从配置中定义的各种可选值列表（如decay_choices, universe_choices等）中
        为每个相关参数随机选择一个值。

        返回:
            Dict[str, Any]: 一个包含随机选择的模拟参数的字典。
        """
        params = {}
        # 为配置中定义为“基因”的参数随机选择值
        if 'decay' in self.ga_genes and self.decay_choices:
            params['decay'] = random.choice(self.decay_choices)
        if 'universe' in self.ga_genes and self.universe_choices:
            params['universe'] = random.choice(self.universe_choices)
        if 'neutralization' in self.ga_genes and self.neutralization_choices:
            params['neutralization'] = random.choice(self.neutralization_choices)
        # 其他可能参与进化的参数也可以按此方式添加

        # 总是包含一些非进化但需要随机选择的基础参数（如果适用）
        if self.delay_choices:
            params.setdefault('delay', random.choice(self.delay_choices)) # setdefault 如果 decay 等已由 ga_genes 设置则不覆盖
        else:
            params.setdefault('delay', 0) # 硬编码默认值

        if self.truncation_choices:
             params.setdefault('truncation', random.choice(self.truncation_choices))
        else:
            params.setdefault('truncation', 0.01)


        self.logger.debug(f"生成的随机模拟参数: {params}")
        return params

    def _get_random_datafield(self) -> str:
        """
        从可用的数据字段中随机选择一个，同时排除黑名单中的字段。

        返回:
            str: 随机选择的数据字段名称。如果可用列表为空，则返回默认字段"close"。
        """
        all_datafields: List[str] = self.config_manager.get("datafields", []) # 假设datafields直接是字符串列表
        if not isinstance(all_datafields, list) or not all(isinstance(df, str) for df in all_datafields):
            self.logger.warning("配置中的 'datafields' 不是预期的字符串列表格式。返回默认 'close'。")
            return "close"

        # 黑名单已在 __init__ 中预处理为 self._processed_blacklist_names (set)

        available_fields = [df for df in all_datafields if df not in self._processed_blacklist_names]

        if not available_fields:
            self.logger.warning("过滤黑名单后，没有可用的数据字段。将返回默认字段 'close'。")
            # 如果所有字段都被列入黑名单，或者原始列表为空，这可能是一个问题
            # 但为了鲁棒性，返回一个最常用的字段
            return "close"

        selected_field = random.choice(available_fields)
        self.logger.debug(f"随机选择的数据字段: {selected_field} (从{len(available_fields)}个可用字段中选择)")
        return selected_field

    def _get_random_operator(self, op_type: str) -> str:
        """
        从指定类型的操作符列表中随机选择一个，同时排除黑名单中的操作符。

        参数:
            op_type (str): 操作符类型，例如 "ts_ops", "binary_ops", "unary_ops"。
                           这对应于ConfigManager中operator_sets下的键名。

        返回:
            str: 随机选择的操作符名称。

        Raises:
            ValueError: 如果指定类型的操作符列表（过滤黑名单后）为空，
                        并且原始列表也为空，无法选择任何操作符。
        """
        op_list_key = f"operator_sets.{op_type}"
        original_ops: List[str] = self.config_manager.get(op_list_key, [])

        if not isinstance(original_ops, list) or not all(isinstance(op, str) for op in original_ops):
             self.logger.error(f"配置中的 '{op_list_key}' 不是预期的字符串列表格式。")
             raise ValueError(f"类型 {op_type} 的操作符列表配置无效。")

        if not original_ops:
            self.logger.error(f"类型 '{op_type}' 的原始操作符列表为空。无法选择操作符。")
            raise ValueError(f"类型 {op_type} 的操作符列表为空，无法选择操作符。")

        # 黑名单已在 __init__ 中预处理为 self._processed_blacklist_names (set)
        available_ops = [op for op in original_ops if op not in self._processed_blacklist_names]

        if not available_ops:
            self.logger.warning(
                f"过滤黑名单后，类型 '{op_type}' 没有可用的操作符。将尝试从原始列表中选择 (可能再次遇到问题)。"
                f"原始列表数量: {len(original_ops)}, 黑名单数量: {len(self._processed_blacklist_names)}"
            )
            # 作为备用，从原始（未过滤）列表中选择，但记录警告
            selected_op = random.choice(original_ops)
            self.logger.debug(f"从原始列表随机选择的操作符 (类型 {op_type}): {selected_op}")
            return selected_op

        selected_op = random.choice(available_ops)
        self.logger.debug(f"随机选择的操作符 (类型 {op_type}): {selected_op} (从{len(available_ops)}个可用操作符中选择)")
        return selected_op

    def generate_alpha_from_template(self) -> Individual:
        """
        使用预定义的模板随机生成一个Alpha表达式个体。

        它从配置中选择一个模板，然后用随机选择的数据字段、操作符或参数值
        填充模板中的占位符。

        返回:
            Individual: 一个新的Alpha个体实例。

        Raises:
            ValueError: 如果模板列表为空、模板格式不正确、占位符无法填充，
                        或生成的表达式无法解析为AST。
        """
        self.logger.info("开始从模板生成Alpha...")
        templates: List[Dict[str, Any]] = self.config_manager.get("alpha_templates", [])
        if not templates:
            self.logger.error("Alpha模板列表为空或未配置。无法从模板生成。")
            raise ValueError("Alpha模板列表为空或未配置。")

        chosen_template_config = random.choice(templates)
        template_str: str = chosen_template_config.get('template')
        data_space: Dict[str, Any] = chosen_template_config.get('data_space', {})
        shared_space: Dict[str, Any] = chosen_template_config.get('shared_space', {})

        if not template_str:
            self.logger.error(f"选中的模板配置不包含'template'字符串: {chosen_template_config}")
            raise ValueError("选中的模板配置无效，缺少'template'字符串。")

        self.logger.debug(f"选中模板: '{template_str}', data_space: {data_space}, shared_space: {shared_space}")

        filled_template_str = template_str

        # 定义替换逻辑的内部函数
        def get_replacement_value(placeholder_name: str, space_definition: Any) -> str:
            if isinstance(space_definition, list): # 直接的值列表
                if not space_definition:
                    self.logger.warning(f"占位符 '{placeholder_name}' 的值列表为空。")
                    return f"<UNFILLED_{placeholder_name}>" # 返回一个可识别的未填充标记
                return str(random.choice(space_definition))
            elif isinstance(space_definition, str): # 类别引用
                category = space_definition.upper() # 转大写以匹配约定
                if category == "DATAFIELD":
                    return self._get_random_datafield()
                elif category.startswith("OPERATOR_"): # e.g., OPERATOR_TS, OPERATOR_BINARY
                    op_type_key = category.split("OPERATOR_", 1)[1].lower() + "_ops" # ts_ops, binary_ops
                    return self._get_random_operator(op_type_key)
                elif category == "TS_OPS_VALUE":
                    if not self.ts_ops_values:
                         self.logger.warning("ts_ops_values 列表为空，无法为占位符提供值。")
                         return f"<UNFILLED_TS_OPS_VALUE_{placeholder_name}>"
                    return str(random.choice(self.ts_ops_values))
                else:
                    self.logger.warning(f"占位符 '{placeholder_name}' 的类别 '{category}' 未知。")
                    return f"<UNKNOWN_CAT_{placeholder_name}>"
            else:
                self.logger.warning(f"占位符 '{placeholder_name}' 的定义格式无法识别: {space_definition}")
                return f"<INVALID_DEF_{placeholder_name}>"

        # 使用re.sub和回调函数进行替换
        def replace_match(match_obj: re.Match) -> str:
            placeholder_key = match_obj.group(1) # 提取占位符名称，例如 <KEY> -> KEY
            self.logger.debug(f"尝试替换占位符: <{placeholder_key}>")

            if placeholder_key in data_space:
                return get_replacement_value(placeholder_key, data_space[placeholder_key])
            elif placeholder_key in shared_space:
                return get_replacement_value(placeholder_key, shared_space[placeholder_key])
            else:
                self.logger.warning(f"在data_space或shared_space中未找到占位符 '<{placeholder_key}>' 的定义。")
                return match_obj.group(0) # 如果未定义，则保持原样

        filled_template_str = re.sub(r"<([\w_]+)>", replace_match, template_str)
        self.logger.info(f"填充后的模板字符串: '{filled_template_str}'")

        # 检查是否还有未替换的占位符
        remaining_placeholders = re.findall(r"<[^>]+>", filled_template_str)
        if remaining_placeholders:
            self.logger.error(f"模板填充后仍存在未处理的占位符: {remaining_placeholders} in '{filled_template_str}'")
            raise ValueError(f"模板填充不完整，存在未处理的占位符: {remaining_placeholders}")

        # AST转换
        expression_ast = parse_expression_to_ast(filled_template_str)
        if expression_ast is None:
            self.logger.error(f"无法将填充后的模板字符串 '{filled_template_str}' 解析为AST。")
            raise ValueError(f"填充后的模板字符串无法解析为AST: '{filled_template_str}'")

        # 生成模拟参数
        sim_params = self._generate_random_simulation_params()

        self.logger.info(f"成功从模板生成Alpha: AST根='{expression_ast.value}', Params={sim_params}")
        return Individual(expression_ast=expression_ast, simulation_params=sim_params)

    def generate_alpha_structurally(self) -> Individual:
        """
        通过结构化随机组合操作符和数据字段来生成Alpha表达式个体。
        【T4.0 初步实现/占位符】此方法旨在后续通过更复杂的算法（如GP的grow/full方法）
        构建具有特定深度和复杂度的AST。

        当前实现仅为占位符，创建一个非常简单的二元操作结构。

        返回:
            Individual: 一个新的Alpha个体实例。

        Raises:
            ValueError: 如果无法获取必要的操作符或数据字段（例如，列表为空）。
        """
        self.logger.info("开始结构化生成Alpha (初步实现)...")

        try:
            # 随机选择一个二元操作符
            # 注意：如果binary_ops列表为空或所有操作符都在黑名单中，_get_random_operator会抛出ValueError
            op = self._get_random_operator("binary_ops")

            # 随机选择两个数据字段
            field1 = self._get_random_datafield()
            field2 = self._get_random_datafield()

            # 构建一个简单的AST: op(field1, field2)
            ast = Node(op, children=[Node(field1), Node(field2)])
            self.logger.debug(f"结构化生成 (占位符) AST: {op}({field1}, {field2})")

        except ValueError as e:
            self.logger.error(f"结构化生成Alpha时出错 (无法获取操作符或数据字段): {e}")
            # 无法创建有意义的AST，抛出或返回一个标记性错误个体（不推荐）
            # 这里选择重新抛出，因为这是生成过程中的关键失败
            raise ValueError(f"结构化生成Alpha失败，无法获取必要组件: {e}")

        # 生成随机模拟参数
        sim_params = self._generate_random_simulation_params()

        self.logger.info(f"成功结构化生成Alpha (占位符): AST根='{ast.value}', Params={sim_params}")
        return Individual(expression_ast=ast, simulation_params=sim_params)

    def generate_random_individual(self) -> Individual:
        """
        根据配置的Alpha生成模式生成一个新的随机Alpha个体。

        模式可以是 "template_based" (基于模板) 或 "structural_based" (结构化生成)。
        如果配置的模式未知，则默认使用基于模板的生成方式。

        返回:
            Individual: 一个新生成的Alpha个体实例。
        """
        self.logger.debug(f"请求生成随机个体，模式: '{self.alpha_generation_mode}'")

        if self.alpha_generation_mode == "template_based":
            return self.generate_alpha_from_template()
        elif self.alpha_generation_mode == "structural_based": # 笔误修正：应该是 structural_based
            return self.generate_alpha_structurally()
        else:
            self.logger.warning(
                f"未知的Alpha生成模式: '{self.alpha_generation_mode}'。将默认使用基于模板的生成。"
            )
            return self.generate_alpha_from_template()


# --- 主程序入口：测试 AlphaEvolutionEngine 初始化 (保持不变) ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    engine_test_logger = logging.getLogger(__name__ + "._engine_main_")

    class MockConfigManager:
        def __init__(self):
            self._config_data = {
                "ga.population_size": 50, "ga.max_generations": 20, "ga.injection_rate": 0.05,
                "ga.min_population_size": 10, "ga.max_population_size": 100,
                "ga.target_successful_alphas": 10, "ga.alpha_generation_mode": "template_based", # "structural_based" or "template_based"
                "operator_sets.terminal_values": ["0.5", "1"], "operator_sets.ts_ops": ["ts_rank", "ts_delay", "ts_max"],
                "operator_sets.binary_ops": ["add", "subtract", "multiply"], "operator_sets.unary_ops": ["abs", "log", "negate"],
                "operator_sets.ts_ops_values": [5, 10, 20], "operator_sets.group_ops": ["group_neutralize"],
                "operator_sets.decay_choices": [0, 1, 5], "operator_sets.delay_choices": [0, 1],
                "operator_sets.truncation_choices": [0.01], "operator_sets.universe_choices": ["TOP500", "TOP1000"],
                "operator_sets.neutralization_choices": [None, "market"],
                "ga.genes": ['decay', 'universe'],
                "datafields": ["open", "close", "high", "low", "volume", "vwap", "cap"],
                "blacklist": [{'name': 'ts_max', 'reason': 'API_ERROR', 'timestamp': '...'}], # Example blacklist
                "alpha_templates": [
                    {
                        "template": "<OPERATOR_BINARY>(<DATAFIELD_A>, <DATAFIELD_B>)",
                        "data_space": {"DATAFIELD_A": "DATAFIELD", "DATAFIELD_B": "DATAFIELD"},
                        "shared_space": {"OPERATOR_BINARY": "OPERATOR_BINARY"}
                    },
                    {
                        "template": "<OPERATOR_UNARY>(<OPERATOR_TS>(<DATAFIELD_C>, <LAG_PARAM>))",
                        "data_space": {"DATAFIELD_C": "DATAFIELD"},
                        "shared_space": {"OPERATOR_UNARY": "OPERATOR_UNARY", "OPERATOR_TS": "OPERATOR_TS", "LAG_PARAM": "TS_OPS_VALUE"}
                    },
                    {
                        "template": "my_specific_alpha_without_placeholders" # Example of a fixed template
                    }
                ]
            }
        def get(self, key_path: str, default: Any = None) -> Any:
            if key_path in self._config_data: return self._config_data[key_path]
            # Simulate dot-path access for nested keys if necessary for other tests, but direct access is fine for this structure
            parts = key_path.split('.')
            current_level = self._config_data
            try:
                for part in parts: current_level = current_level[part]
                return current_level
            except (KeyError, TypeError):
                engine_test_logger.warning(f"MockConfigManager: Key '{key_path}' not found, returning default '{default}'")
                return default
        def set_blacklist(self, data: List[Any]): engine_test_logger.debug(f"MockConfigManager.set_blacklist called with {data}")
        def set_datafields(self, data: List[Any]): engine_test_logger.debug(f"MockConfigManager.set_datafields called with {data}")

    class MockWQBAPIClient: # type: ignore
        def __init__(self, config_manager: ConfigManager): self.config_manager = config_manager; engine_test_logger.info("MockWQBAPIClient initialized.")
    class MockResultHandler: # type: ignore
        def __init__(self, config_manager: ConfigManager, wqb_api_client: WQB_API_Client, alpha_evolution_engine_ref=None):
            self.config_manager = config_manager; self.wqb_api_client = wqb_api_client; self.alpha_evolution_engine_ref = alpha_evolution_engine_ref; engine_test_logger.info("MockResultHandler initialized.")

    engine_test_logger.info("开始测试 AlphaEvolutionEngine 初始化及Alpha生成...")
    try:
        mock_config = MockConfigManager()
        mock_wqb_client = MockWQBAPIClient(mock_config)
        mock_result_handler = MockResultHandler(mock_config, mock_wqb_client)

        engine = AlphaEvolutionEngine(
            config_manager=mock_config, # type: ignore
            wqb_api_client=mock_wqb_client, # type: ignore
            result_handler=mock_result_handler # type: ignore
        )
        engine_test_logger.info(f"AlphaEvolutionEngine 初始化成功。")

        engine_test_logger.info("\n--- 测试 _generate_random_simulation_params ---")
        for i in range(3):
            params = engine._generate_random_simulation_params()
            engine_test_logger.info(f"随机模拟参数 {i+1}: {params}")
            assert 'decay' in params and params['decay'] in mock_config.get('operator_sets.decay_choices')
            assert 'universe' in params and params['universe'] in mock_config.get('operator_sets.universe_choices')

        engine_test_logger.info("\n--- 测试 _get_random_datafield ---")
        for i in range(5): # Test multiple times due to randomness
            df = engine._get_random_datafield()
            engine_test_logger.info(f"随机数据字段 {i+1}: {df}")
            assert df in mock_config.get('datafields')
            assert df not in [item['name'] for item in mock_config.get('blacklist')] # Ensure not blacklisted

        engine_test_logger.info("\n--- 测试 _get_random_operator ---")
        for i in range(3):
            op_ts = engine._get_random_operator("ts_ops")
            engine_test_logger.info(f"随机TS操作符 {i+1}: {op_ts}")
            assert op_ts in mock_config.get('operator_sets.ts_ops')
            assert op_ts != "ts_max" # ts_max is in blacklist in mock_config

            op_bin = engine._get_random_operator("binary_ops")
            engine_test_logger.info(f"随机二元操作符 {i+1}: {op_bin}")
            assert op_bin in mock_config.get('operator_sets.binary_ops')

        engine_test_logger.info("\n--- 测试 generate_alpha_from_template ---")
        for i in range(3):
            try:
                individual_template = engine.generate_alpha_from_template()
                engine_test_logger.info(f"模板生成个体 {i+1}: {individual_template.get_expression_str()}, Params: {individual_template.simulation_params}")
                assert individual_template.expression_ast is not None
                assert "<" not in individual_template.get_expression_str() # Check for unfilled placeholders
            except ValueError as ve:
                engine_test_logger.error(f"模板生成个体 {i+1} 失败: {ve}")


        engine_test_logger.info("\n--- 测试 generate_alpha_structurally (占位符) ---")
        try:
            individual_struct = engine.generate_alpha_structurally()
            engine_test_logger.info(f"结构化生成个体 (占位符): {individual_struct.get_expression_str()}, Params: {individual_struct.simulation_params}")
            assert individual_struct.expression_ast is not None
            assert individual_struct.expression_ast.value in mock_config.get('operator_sets.binary_ops')
        except ValueError as ve:
            engine_test_logger.error(f"结构化生成个体失败: {ve}")


        engine_test_logger.info("\n--- 测试 generate_random_individual (template mode) ---")
        engine.alpha_generation_mode = "template_based"
        individual_random_template = engine.generate_random_individual()
        engine_test_logger.info(f"随机个体 (模板模式): {individual_random_template.get_expression_str()}")
        assert "<" not in individual_random_template.get_expression_str()


        engine_test_logger.info("\n--- 测试 generate_random_individual (structural mode) ---")
        engine.alpha_generation_mode = "structural_based" # Corrected typo from "structural_based"
        individual_random_struct = engine.generate_random_individual()
        engine_test_logger.info(f"随机个体 (结构化模式): {individual_random_struct.get_expression_str()}")
        assert individual_random_struct.expression_ast.value in mock_config.get('operator_sets.binary_ops')

        engine_test_logger.info("Alpha生成方法测试完成。")

    except Exception as e:
        engine_test_logger.critical(f"测试 AlphaEvolutionEngine Alpha生成时发生错误: {e}", exc_info=True)

