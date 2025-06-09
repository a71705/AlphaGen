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
            self.logger.warning(f"当前datafields配置值: {all_datafields}")
            self.logger.warning(f"datafields类型: {type(all_datafields)}")
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
             self.logger.error(f"当前{op_list_key}配置值: {original_ops}")
             self.logger.error(f"{op_list_key}类型: {type(original_ops)}")
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
            self.logger.error(f"当前配置中的alpha_templates值: {templates}")
            self.logger.error("请检查config/alpha_templates.json文件是否存在且格式正确")
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

    def run_optimization(self, resume: bool = False) -> None:
        """
        运行遗传算法优化过程，搜索最优的Alpha因子。
        
        该方法是Alpha挖掘的主要入口点，负责执行完整的遗传算法流程，
        包括种群初始化、进化迭代、适应度评估和结果保存。
        
        参数:
            resume (bool): 是否从上次保存的进度恢复。如果为True，
                          将尝试加载之前的种群和进化状态；
                          如果为False，将开始全新的优化过程。
        
        Raises:
            Exception: 如果优化过程中发生严重错误。
        """
        self.logger.info(f"开始Alpha因子遗传算法优化 (resume={resume})...")
        
        try:
            # 获取遗传算法配置参数
            population_size = self.config_manager.get('ga.population_size', 50)
            max_generations = self.config_manager.get('ga.max_generations', 100)
            target_successful_alphas = self.config_manager.get('ga.target_successful_alphas', 10)
            
            self.logger.info(f"GA配置: 种群大小={population_size}, 最大代数={max_generations}, 目标成功Alpha数={target_successful_alphas}")
            
            # 初始化或恢复种群
            if resume:
                # 使用 ResultHandler 加载 GA 状态
                ga_state = self.result_handler.load_ga_state()
                if ga_state and 'population' in ga_state:
                    self.current_population = ga_state['population']
                    self.current_generation = ga_state.get('current_generation', 0)
                    self.total_successful_alphas_found = ga_state.get('total_successful_alphas_found', 0)
                    self.logger.info(f"成功从 ResultHandler 恢复种群，当前代数: {self.current_generation}")
                else:
                    self.logger.warning("无法从 ResultHandler 恢复种群，将开始新的优化过程")
                    self.current_population = self._initialize_population(population_size)
                    self.current_generation = 0
                    self.total_successful_alphas_found = 0
            else:
                self.logger.info("开始新的优化过程，初始化种群...")
                self.current_population = self._initialize_population(population_size)
                self.current_generation = 0
                self.total_successful_alphas_found = 0
            
            # 主要的进化循环
            for generation in range(self.current_generation, max_generations):
                self.logger.info(f"\n=== 第 {generation + 1}/{max_generations} 代进化开始 ===")
                
                # 使用替补式并发评估框架评估当前种群
                self.logger.info("开始并发评估种群适应度...")
                evaluation_results = self._evaluate_population_with_replenishing_concurrency()
                
                # 更新种群中个体的适应度和状态
                qualified_count = 0
                for individual, is_qualified in evaluation_results:
                    if is_qualified:
                        qualified_count += 1
                        self.total_successful_alphas_found += 1
                
                self.logger.info(f"第 {generation + 1} 代评估完成: {qualified_count}/{len(self.current_population)} 个合格Alpha")
                
                # 自适应调整种群大小和遗传参数
                self._adjust_population_size(qualified_count)
                
                # 检查终止条件
                # 1. 检查是否达到目标成功Alpha数量
                if self.total_successful_alphas_found >= target_successful_alphas:
                    self.logger.info(f"已达到目标成功Alpha数量 ({self.total_successful_alphas_found}/{target_successful_alphas})，优化成功完成")
                    break
                
                # 2. 检查是否达到最大代数（在循环条件中已经处理，这里添加日志）
                if generation >= max_generations - 1:
                    self.logger.info(f"已达到最大代数 ({max_generations})，优化结束")
                    break
                
                # 更新当前代数
                self.current_generation = generation + 1
                
                # 保存当前 GA 状态
                ga_state = {
                    'population': self.current_population,
                    'current_generation': self.current_generation,
                    'total_successful_alphas_found': self.total_successful_alphas_found
                }
                self.result_handler.save_ga_state(ga_state)
                
                # 如果不是最后一代，进行进化操作
                if generation < max_generations - 1:
                    self.logger.info("执行进化操作...")
                    # 将评估结果转换为 (Individual, bool) 格式，bool表示是否合格
                    evaluated_population = [(ind, ind.fitness_score >= self.config_manager.get('ga.fitness_function.min_acceptable_fitness', 0.5)) for ind in self.current_population]
                    self.current_population = self._generate_next_generation(evaluated_population)
                
                self.logger.info(f"第 {generation + 1} 代进化完成")
            
            # 优化完成
            self.logger.info(f"\n=== 优化完成 ===")
            self.logger.info(f"总共发现 {self.total_successful_alphas_found} 个成功Alpha")
            
            self.logger.info("Alpha因子遗传算法优化完成")
            
        except KeyboardInterrupt:
            self.logger.warning("优化过程被用户中断")
            # 保存当前 GA 状态
            ga_state = {
                'population': self.current_population,
                'current_generation': self.current_generation,
                'total_successful_alphas_found': self.total_successful_alphas_found
            }
            self.result_handler.save_ga_state(ga_state)
            self.logger.info("当前进度已保存")
            raise
        except Exception as e:
            self.logger.error(f"优化过程中发生错误: {e}", exc_info=True)
            raise
    
    def _initialize_population(self, size: int) -> List[Individual]:
        """
        初始化遗传算法种群。
        
        参数:
            size (int): 种群大小
            
        返回:
            List[Individual]: 初始化的种群
        """
        self.logger.info(f"初始化种群，大小: {size}")
        population = []
        
        for i in range(size):
            try:
                individual = self.generate_random_individual()
                population.append(individual)
                self.logger.debug(f"生成个体 {i+1}/{size}: {individual.get_expression_str()}")
            except Exception as e:
                self.logger.error(f"生成第 {i+1} 个个体时发生错误: {e}")
                self.logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
                # 继续尝试生成其他个体
                continue
        
        self.logger.info(f"成功初始化种群，实际大小: {len(population)}")
        return population
    
    def _evaluate_population_fitness(self, population: List[Individual]) -> None:
        """
        评估种群中所有个体的适应度。
        
        参数:
            population (List[Individual]): 待评估的种群
        """
        self.logger.info(f"开始评估 {len(population)} 个个体的适应度")
        
        for i, individual in enumerate(population):
            try:
                # 这里应该调用WQB API进行Alpha测试
                # 目前作为占位符实现
                fitness_score = self._calculate_individual_fitness(individual)
                individual.fitness_score = fitness_score
                self.logger.debug(f"个体 {i+1} 适应度: {fitness_score:.4f}")
            except Exception as e:
                self.logger.error(f"评估个体 {i+1} 适应度时发生错误: {e}")
                individual.fitness_score = 0.0  # 设置默认适应度
    
    def _calculate_individual_fitness(self, individual: Individual) -> float:
        """
        计算单个个体的适应度分数。
        
        参数:
            individual (Individual): 待评估的个体
            
        返回:
            float: 适应度分数
        """
        # 占位符实现 - 实际应该通过WQB API测试Alpha性能
        # 这里返回一个随机分数作为示例
        import random
        fitness = random.uniform(0.0, 1.0)
        self.logger.debug(f"计算个体适应度 (占位符): {fitness:.4f}")
        return fitness
    
    def _evaluate_population_with_replenishing_concurrency(self) -> List[tuple[Individual, bool]]:
        """
        使用替补式并发调度逻辑评估当前种群的适应度。
        
        该方法是遗传算法世代循环中进行所有 Alpha 模拟、监控和结果处理的核心调度器。
        它利用 ThreadPoolExecutor 进行并发处理，使用 WQB_API_Client 提交模拟任务，
        并与 ResultHandler 协作处理每个 Alpha 的结果。
        
        返回:
            List[tuple[Individual, bool]]: 包含 (Individual, is_qualified) 元组的列表，
                                          其中 is_qualified 表示该 Alpha 是否合格
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue
        import time
        
        self.logger.info(f"开始替补式并发评估 {len(self.current_population)} 个个体")
        
        # 获取并发配置
        max_workers = self.config_manager.get("general.max_concurrent_simulations", 5)
        self.logger.info(f"使用 {max_workers} 个并发线程进行评估")
        
        # 初始化任务队列和结果列表
        tasks_to_submit = queue.Queue()
        for individual in self.current_population:
            tasks_to_submit.put(individual)
        
        running_futures = {}  # {Future: Individual}
        evaluation_results = []  # [(Individual, bool)]
        total_successful_alphas = 0
        completed_count = 0
        total_count = len(self.current_population)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 初始提交任务，填满线程池
            while len(running_futures) < max_workers and not tasks_to_submit.empty():
                individual = tasks_to_submit.get()
                future = executor.submit(self._simulate_and_process_individual, individual)
                running_futures[future] = individual
                self.logger.debug(f"提交个体任务: {individual.get_expression_str()[:50]}...")
            
            # 主循环：处理完成的任务并补充新任务
            while running_futures:
                # 等待至少一个任务完成
                for future in as_completed(running_futures):
                    individual = running_futures.pop(future)
                    completed_count += 1
                    
                    try:
                        # 获取任务结果
                        result = future.result()
                        individual_result, is_qualified = result
                        
                        # 更新个体信息
                        individual.fitness_score = individual_result.fitness_score
                        individual.alpha_id = individual_result.alpha_id
                        
                        # 记录结果
                        evaluation_results.append((individual, is_qualified))
                        
                        if is_qualified:
                            total_successful_alphas += 1
                            self.logger.info(
                                f"✓ 个体 {completed_count}/{total_count} 合格 - "
                                f"Alpha ID: {individual.alpha_id}, 适应度: {individual.fitness_score:.4f}"
                            )
                        else:
                            self.logger.debug(
                                f"✗ 个体 {completed_count}/{total_count} 不合格 - "
                                f"适应度: {individual.fitness_score:.4f}"
                            )
                        
                        # 打印进度
                        if completed_count % 10 == 0 or completed_count == total_count:
                            progress_percent = (completed_count / total_count) * 100
                            self.logger.info(
                                f"评估进度: {completed_count}/{total_count} ({progress_percent:.1f}%) - "
                                f"合格: {total_successful_alphas}"
                            )
                    
                    except Exception as e:
                        # 处理任务异常
                        self.logger.error(f"个体评估任务失败: {e}", exc_info=True)
                        
                        # 为失败的个体分配极低适应度
                        individual.fitness_score = float('-inf')
                        individual.alpha_id = None
                        evaluation_results.append((individual, False))
                    
                    # 补充新任务（如果还有待处理的个体）
                    if not tasks_to_submit.empty():
                        new_individual = tasks_to_submit.get()
                        new_future = executor.submit(self._simulate_and_process_individual, new_individual)
                        running_futures[new_future] = new_individual
                        self.logger.debug(f"补充新任务: {new_individual.get_expression_str()[:50]}...")
                    
                    # 只处理一个完成的任务，然后重新开始循环
                    break
        
        self.logger.info(
            f"并发评估完成 - 总计: {total_count}, 合格: {total_successful_alphas} "
            f"({(total_successful_alphas/total_count)*100:.1f}%)"
        )
        
        return evaluation_results
    
    def _simulate_and_process_individual(self, individual: Individual) -> tuple[Individual, bool]:
        """
        对单个个体进行模拟和结果处理的完整流程。
        
        该方法封装了单个 Alpha 的完整评估流程：
        1. 构建模拟请求载荷
        2. 提交到 WQB API 进行模拟
        3. 监控模拟进度
        4. 使用 ResultHandler 处理结果
        
        参数:
            individual (Individual): 待评估的个体
            
        返回:
            tuple[Individual, bool]: (更新后的个体, 是否合格)
        """
        try:
            # 1. 构建模拟请求载荷
            simulation_payload = self._build_simulation_payload(individual)
            self.logger.debug(f"构建模拟载荷完成: {individual.get_expression_str()[:50]}...")
            
            # 2. 提交模拟任务到 WQB API
            submission_result = self.wqb_api_client.submit_alpha_for_simulation(simulation_payload)
            if not submission_result or 'simulation_id' not in submission_result:
                raise ValueError("模拟提交失败：未获得有效的 simulation_id")
            
            simulation_id = submission_result['simulation_id']
            self.logger.debug(f"模拟任务已提交，ID: {simulation_id}")
            
            # 3. 监控模拟进度直到完成
            simulation_result = self.wqb_api_client.monitor_simulation_progress(simulation_id)
            if not simulation_result:
                raise ValueError(f"模拟监控失败：simulation_id {simulation_id}")
            
            # 检查模拟是否成功完成
            if not simulation_result.get('success', False):
                # 模拟失败，可能是由于不可访问操作符等问题
                error_message = simulation_result.get('message', '未知错误')
                status = simulation_result.get('status', 'UNKNOWN')
                self.logger.warning(f"模拟失败，跳过后续处理: {individual.get_expression_str()[:50]}... - {error_message}")
                
                # 为失败的个体分配极低适应度，直接返回
                individual.fitness_score = float('-inf')
                individual.alpha_id = None
                return individual, False
            
            self.logger.debug(f"模拟完成，ID: {simulation_id}")
            
            # 4. 使用 ResultHandler 处理模拟结果
            alpha_id = simulation_result.get('simulation_id')
            original_payload = {
                'regular': individual.get_expression_str(),
                'settings': simulation_result.get('simulation_settings', {})
            }
            wqb_raw_result = simulation_result.get('response_data', {})
            
            is_qualified, fitness_score, processing_result = self.result_handler.handle_simulation_result(
                alpha_id,
                original_payload,
                wqb_raw_result
            )
            
            # 5. 解析处理结果（handle_simulation_result现在返回元组）
            # is_qualified, fitness_score, processing_result 已经从上面的调用中获得
            alpha_id = processing_result.get('alpha_id')
            
            # 6. 更新个体信息
            individual.fitness_score = fitness_score
            individual.alpha_id = alpha_id
            
            return individual, is_qualified
            
        except Exception as e:
            # 处理模拟或处理过程中的异常
            self.logger.error(
                f"个体模拟处理失败: {individual.get_expression_str()[:50]}... - {e}", 
                exc_info=True
            )
            
            # 为失败的个体分配极低适应度
            individual.fitness_score = float('-inf')
            individual.alpha_id = None
            
            return individual, False
    
    def _build_simulation_payload(self, individual: Individual) -> Dict[str, Any]:
        """
        为个体构建 WQB API 模拟请求的载荷。
        
        参数:
            individual (Individual): 需要构建载荷的个体
            
        返回:
            Dict[str, Any]: WQB API 模拟请求载荷
        """
        try:
            # 获取个体的表达式字符串
            expression_str = individual.get_expression_str()
            
            # 构建基础载荷
            payload = {
                'expression': expression_str,
                'simulation_params': individual.simulation_params.copy()
            }
            
            # 添加默认参数（如果个体参数中没有指定）
            default_params = {
                'decay': 0,
                'delay': 1,
                'truncation': 0.01,
                'universe': 'TOP3000',
                'neutralization': None
            }
            
            for key, default_value in default_params.items():
                if key not in payload['simulation_params']:
                    payload['simulation_params'][key] = default_value
            
            self.logger.debug(f"构建模拟载荷: {expression_str[:50]}...")
            return payload
            
        except Exception as e:
            self.logger.error(f"构建模拟载荷失败: {e}", exc_info=True)
            raise ValueError(f"无法为个体构建模拟载荷: {e}")
    
    def _extract_successful_alphas(self, population: List[Individual]) -> List[Individual]:
        """
        从种群中提取成功的Alpha个体。
        
        参数:
            population (List[Individual]): 当前种群
            
        返回:
            List[Individual]: 成功的Alpha个体列表
        """
        min_fitness = self.config_manager.get('ga.fitness_function.min_acceptable_fitness', 0.7)
        successful = [ind for ind in population if ind.fitness_score >= min_fitness]
        
        if successful:
            self.logger.info(f"发现 {len(successful)} 个成功Alpha (适应度 >= {min_fitness})")
        
        return successful
    
    def _generate_next_generation(self, evaluated_population: List[tuple[Individual, bool]]) -> List[Individual]:
        """
        根据当前代的评估结果生成下一代种群。
        实现精英选择、交叉、变异和多样性注入机制。
        
        参数:
            evaluated_population (List[tuple[Individual, bool]]): 当前种群的评估结果，
                                                                  每个元组包含个体和是否合格的布尔值
            
        返回:
            List[Individual]: 下一代种群
        """
        self.logger.info("开始生成下一代种群")
        
        # 获取配置参数
        elite_count = self.config_manager.get('ga.elitism_size', 2)
        crossover_rate = self.config_manager.get('ga.crossover_probability', 0.7)
        mutation_rate = self.config_manager.get('ga.mutation_probability', 0.2)
        injection_rate = self.config_manager.get('ga.injection_rate', 0.1)
        population_size = self.config_manager.get('ga.population_size', 100)
        
        # 分离合格和不合格个体
        qualified_individuals = [ind for ind, is_qualified in evaluated_population if is_qualified]
        all_individuals = [ind for ind, _ in evaluated_population]
        
        # 按适应度排序
        qualified_individuals.sort(key=lambda x: x.fitness_score, reverse=True)
        all_individuals.sort(key=lambda x: x.fitness_score, reverse=True)
        
        next_generation = []
        
        # 1. 精英选择：保留最优的合格个体
        if qualified_individuals:
            elite_size = min(elite_count, len(qualified_individuals))
            elites = qualified_individuals[:elite_size]
            next_generation.extend(elites)
            self.logger.info(f"精英选择：保留 {elite_size} 个最优个体")
        else:
            # 如果没有合格个体，从所有个体中选择精英
            elite_size = min(elite_count, len(all_individuals))
            elites = all_individuals[:elite_size]
            next_generation.extend(elites)
            self.logger.warning(f"无合格个体，从所有个体中选择 {elite_size} 个精英")
        
        # 2. 生成剩余个体
        while len(next_generation) < population_size:
            try:
                # 决定是否进行多样性注入
                if random.random() < injection_rate:
                    # 多样性注入：生成全新的随机个体
                    new_individual = self.generate_random_individual()
                    next_generation.append(new_individual)
                    self.logger.debug("注入新的随机个体")
                else:
                    # 选择父代进行交叉和变异
                    parent_pool = qualified_individuals if qualified_individuals else all_individuals
                    if len(parent_pool) >= 2:
                        parent1, parent2 = self._select_parents(parent_pool)
                        
                        # 交叉操作
                        if random.random() < crossover_rate:
                            child = self._crossover(parent1, parent2)
                            self.logger.debug("执行交叉操作")
                        else:
                            # 如果不交叉，随机选择一个父代作为基础
                            child = self._copy_individual(random.choice([parent1, parent2]))
                        
                        # 变异操作
                        if random.random() < mutation_rate:
                            child = self._mutate(child)
                            self.logger.debug("执行变异操作")
                        
                        next_generation.append(child)
                    else:
                        # 父代不足，生成随机个体
                        new_individual = self.generate_random_individual()
                        next_generation.append(new_individual)
                        
            except Exception as e:
                self.logger.error(f"生成下一代个体时发生错误: {e}")
                self.logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.debug(f"完整错误堆栈: {traceback.format_exc()}")
                # 发生错误时生成随机个体作为备选
                try:
                    backup_individual = self.generate_random_individual()
                    next_generation.append(backup_individual)
                except Exception as backup_e:
                    self.logger.error(f"生成备选个体也失败: {backup_e}")
                    break
        
        # 确保种群大小正确
        if len(next_generation) > population_size:
            next_generation = next_generation[:population_size]
        elif len(next_generation) < population_size:
            # 补充随机个体
            while len(next_generation) < population_size:
                try:
                    supplement_individual = self.generate_random_individual()
                    next_generation.append(supplement_individual)
                except Exception as e:
                    self.logger.error(f"补充个体失败: {e}")
                    break
        
        self.logger.info(f"下一代种群生成完成，大小: {len(next_generation)}")
        return next_generation
    
    def _select_parents(self, parent_pool: List[Individual]) -> tuple[Individual, Individual]:
        """
        从父代池中选择两个父代进行繁殖。
        使用锦标赛选择法。
        
        参数:
            parent_pool (List[Individual]): 可选择的父代池
            
        返回:
            tuple[Individual, Individual]: 选中的两个父代
        """
        tournament_size = self.config_manager.get('ga.tournament_size', 3)
        tournament_size = min(tournament_size, len(parent_pool))
        
        def tournament_selection() -> Individual:
            """锦标赛选择一个个体"""
            tournament = random.sample(parent_pool, tournament_size)
            return max(tournament, key=lambda x: x.fitness_score)
        
        parent1 = tournament_selection()
        parent2 = tournament_selection()
        
        # 确保两个父代不是同一个对象
        max_attempts = 10
        attempts = 0
        while parent1 is parent2 and len(parent_pool) > 1 and attempts < max_attempts:
            parent2 = tournament_selection()
            attempts += 1
        
        return parent1, parent2
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        对两个父代进行交叉操作，生成子代。
        
        参数:
            parent1 (Individual): 第一个父代
            parent2 (Individual): 第二个父代
            
        返回:
            Individual: 交叉后的子代
        """
        self.logger.debug(f"执行交叉操作: 父代1适应度={parent1.fitness_score:.4f}, 父代2适应度={parent2.fitness_score:.4f}")
        
        # 参数交叉：随机选择每个模拟参数来自哪个父代
        genes = self.config_manager.get('ga.genes', ['decay', 'universe', 'neutralization'])
        child_sim_params = {}
        
        for gene in genes:
            if gene in parent1.simulation_params and gene in parent2.simulation_params:
                # 随机选择来自父代1或父代2的参数值
                if random.random() < 0.5:
                    child_sim_params[gene] = parent1.simulation_params[gene]
                else:
                    child_sim_params[gene] = parent2.simulation_params[gene]
            elif gene in parent1.simulation_params:
                child_sim_params[gene] = parent1.simulation_params[gene]
            elif gene in parent2.simulation_params:
                child_sim_params[gene] = parent2.simulation_params[gene]
        
        # 复制其他参数（非基因参数）
        for key, value in parent1.simulation_params.items():
            if key not in genes and key not in child_sim_params:
                child_sim_params[key] = value
        
        # 结构交叉（AST交叉）
        child_ast = None
        alpha_generation_mode = self.config_manager.get('ga.alpha_generation_mode', 'template_based')
        
        if alpha_generation_mode == 'structural_based':
            # 在结构模式下，尝试进行AST交叉
            # 注意：这里调用utils.py中的AST功能，即使它们可能是占位符实现
            try:
                from .utils import parse_expression_to_ast, generate_expression_from_ast
                
                # 简单的AST交叉：随机选择一个父代的AST作为基础
                # 更复杂的交叉可以在utils.py中实现专门的AST交叉函数
                if random.random() < 0.5:
                    child_ast = parent1.expression_ast
                else:
                    child_ast = parent2.expression_ast
                
                self.logger.debug("结构模式下执行AST交叉（当前为简单选择）")
                
                # TODO: 在utils.py中实现更复杂的AST交叉算法
                # 例如：child_ast = crossover_ast(parent1.expression_ast, parent2.expression_ast)
                
            except ImportError as e:
                self.logger.warning(f"无法导入AST工具函数，使用父代1的AST: {e}")
                child_ast = parent1.expression_ast
            except Exception as e:
                self.logger.error(f"AST交叉过程中发生错误: {e}")
                child_ast = parent1.expression_ast
        else:
            # 在模板模式下，随机选择一个父代的AST
            child_ast = parent1.expression_ast if random.random() < 0.5 else parent2.expression_ast
        
        # 创建子代个体
        child = Individual(
            expression_ast=child_ast,
            simulation_params=child_sim_params
        )
        
        self.logger.debug(f"交叉完成，生成子代")
        return child
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        对个体进行变异操作。
        
        参数:
            individual (Individual): 要变异的个体
            
        返回:
            Individual: 变异后的个体
        """
        self.logger.debug(f"执行变异操作: 个体适应度={individual.fitness_score:.4f}")
        
        # 创建个体的副本进行变异
        mutated_individual = self._copy_individual(individual)
        
        # 参数变异：随机选择一个基因进行变异
        genes = self.config_manager.get('ga.genes', ['decay', 'universe', 'neutralization'])
        available_genes = [gene for gene in genes if gene in mutated_individual.simulation_params]
        
        if available_genes:
            gene_to_mutate = random.choice(available_genes)
            
            # 获取该基因的可选值范围
            gene_options = self._get_gene_options(gene_to_mutate)
            if gene_options:
                new_value = random.choice(gene_options)
                old_value = mutated_individual.simulation_params[gene_to_mutate]
                mutated_individual.simulation_params[gene_to_mutate] = new_value
                self.logger.debug(f"参数变异: {gene_to_mutate} 从 {old_value} 变为 {new_value}")
        
        # 结构变异（AST变异）
        alpha_generation_mode = self.config_manager.get('ga.alpha_generation_mode', 'template_based')
        
        if alpha_generation_mode == 'structural_based':
            # 在结构模式下，尝试进行AST变异
            try:
                from .utils import parse_expression_to_ast, generate_expression_from_ast
                
                # 简单的AST变异：这里是占位符实现
                # 实际的AST变异应该在utils.py中实现专门的函数
                
                self.logger.debug("结构模式下执行AST变异（当前为占位符实现）")
                
                # TODO: 在utils.py中实现AST变异算法
                # 例如：mutated_individual.expression_ast = mutate_ast(mutated_individual.expression_ast)
                # 当前保持AST不变
                
            except ImportError as e:
                self.logger.warning(f"无法导入AST工具函数，跳过AST变异: {e}")
            except Exception as e:
                self.logger.error(f"AST变异过程中发生错误: {e}")
        
        # 重置适应度分数，因为个体已经改变
        mutated_individual.fitness_score = 0.0
        mutated_individual.alpha_id = None
        
        self.logger.debug(f"变异完成")
        return mutated_individual
    
    def _copy_individual(self, individual: Individual) -> Individual:
        """
        创建个体的深度副本。
        
        参数:
            individual (Individual): 要复制的个体
            
        返回:
            Individual: 个体的副本
        """
        import copy
        
        copied_individual = Individual(
            expression_ast=copy.deepcopy(individual.expression_ast),
            simulation_params=copy.deepcopy(individual.simulation_params)
        )
        
        # 复制其他属性
        copied_individual.fitness_score = individual.fitness_score
        copied_individual.alpha_id = individual.alpha_id
        
        return copied_individual
    
    def _get_gene_options(self, gene_name: str) -> List[Any]:
        """
        获取指定基因的可选值范围。
        
        参数:
            gene_name (str): 基因名称
            
        返回:
            List[Any]: 该基因的可选值列表
        """
        # 根据基因名称返回相应的可选值
        # 这些值应该与generate_random_individual中使用的值保持一致
        
        if gene_name == 'decay':
            return list(range(1, 21))  # 1-20
        elif gene_name == 'universe':
            return ['TOP_2000_LIQUID', 'TOP_3000', 'TOP_1000']
        elif gene_name == 'neutralization':
            return ['SUBINDUSTRY', 'INDUSTRY', 'SECTOR', 'NONE']
        else:
            # 对于未知基因，尝试从配置中获取
            gene_config_key = f'ga.gene_options.{gene_name}'
            return self.config_manager.get(gene_config_key, [])
    
    def _adjust_population_size(self, successful_this_gen: int) -> None:
        """
        根据当前代的成功Alpha数量自适应调整种群大小和遗传参数。
        
        参数:
            successful_this_gen (int): 当前代成功发现的合格Alpha数量
        """
        if len(self.current_population) == 0:
            self.logger.warning("当前种群为空，跳过自适应调整")
            return
        
        # 计算当前代的成功率
        success_rate = successful_this_gen / len(self.current_population)
        self.logger.info(f"当前代成功率: {success_rate:.4f} ({successful_this_gen}/{len(self.current_population)})")
        
        # 获取自适应调整配置
        adjust_thresholds = self.config_manager.get('ga.population_adjust_thresholds', {})
        crossover_rules = self.config_manager.get('ga.adaptive_crossover_rate_rules', {})
        mutation_rules = self.config_manager.get('ga.adaptive_mutation_rate_rules', {})
        
        # 获取种群大小限制
        min_pop_size = self.config_manager.get('ga.min_population_size', 20)
        max_pop_size = self.config_manager.get('ga.max_population_size', 200)
        
        # 1. 自适应调整种群大小
        current_pop_size = len(self.current_population)
        new_pop_size = current_pop_size
        
        low_threshold = adjust_thresholds.get('low_success_rate', 0.05)
        high_threshold = adjust_thresholds.get('high_success_rate', 0.20)
        
        if success_rate < low_threshold:
            # 成功率过低，扩大种群
            expand_factor = adjust_thresholds.get('expand_factor_high', 2.0)
            new_pop_size = int(current_pop_size * expand_factor)
            self.logger.info(f"成功率过低 ({success_rate:.4f} < {low_threshold})，扩大种群: {current_pop_size} -> {new_pop_size}")
        elif success_rate < high_threshold:
            # 成功率较低，适度扩大种群
            expand_factor = adjust_thresholds.get('expand_factor_low', 1.5)
            new_pop_size = int(current_pop_size * expand_factor)
            self.logger.info(f"成功率较低 ({success_rate:.4f} < {high_threshold})，适度扩大种群: {current_pop_size} -> {new_pop_size}")
        elif success_rate > high_threshold:
            # 成功率较高，可以收缩种群以提高效率
            shrink_factor = adjust_thresholds.get('shrink_factor', 0.7)
            new_pop_size = int(current_pop_size * shrink_factor)
            self.logger.info(f"成功率较高 ({success_rate:.4f} > {high_threshold})，收缩种群: {current_pop_size} -> {new_pop_size}")
        
        # 确保种群大小在合理范围内
        new_pop_size = max(min_pop_size, min(new_pop_size, max_pop_size))
        
        if new_pop_size != current_pop_size:
            # 更新种群大小配置
            self.config_manager.set_runtime_param("ga", "population_size", new_pop_size)
            self.logger.info(f"种群大小已调整为: {new_pop_size} (限制范围: {min_pop_size}-{max_pop_size})")
        
        # 2. 自适应调整交叉率
        current_crossover_rate = self.config_manager.get('ga.crossover_probability', 0.7)
        new_crossover_rate = current_crossover_rate
        
        crossover_low = crossover_rules.get('low_success_rate', 0.05)
        crossover_medium = crossover_rules.get('medium_success_rate', 0.15)
        crossover_high = crossover_rules.get('high_success_rate', 0.25)
        
        if success_rate < crossover_low:
            # 成功率很低，提高交叉率以增加探索
            new_crossover_rate = crossover_rules.get('rate_for_low', 0.8)
            self.logger.info(f"成功率很低，提高交叉率: {current_crossover_rate:.3f} -> {new_crossover_rate:.3f}")
        elif success_rate < crossover_medium:
            # 成功率中等偏低，使用中等交叉率
            new_crossover_rate = crossover_rules.get('rate_for_medium', 0.7)
            self.logger.info(f"成功率中等偏低，调整交叉率: {current_crossover_rate:.3f} -> {new_crossover_rate:.3f}")
        elif success_rate >= crossover_high:
            # 成功率较高，降低交叉率以保持稳定
            new_crossover_rate = crossover_rules.get('rate_for_high', 0.6)
            self.logger.info(f"成功率较高，降低交叉率: {current_crossover_rate:.3f} -> {new_crossover_rate:.3f}")
        
        if abs(new_crossover_rate - current_crossover_rate) > 0.01:  # 避免微小变化
            self.config_manager.set_runtime_param("ga", "crossover_probability", new_crossover_rate)
            self.logger.info(f"交叉率已调整为: {new_crossover_rate:.3f}")
        
        # 3. 自适应调整变异率
        current_mutation_rate = self.config_manager.get('ga.mutation_probability', 0.2)
        new_mutation_rate = current_mutation_rate
        
        mutation_low = mutation_rules.get('low_success_rate', 0.05)
        mutation_medium = mutation_rules.get('medium_success_rate', 0.15)
        mutation_high = mutation_rules.get('high_success_rate', 0.25)
        
        if success_rate < mutation_low:
            # 成功率很低，提高变异率以增加多样性
            new_mutation_rate = mutation_rules.get('rate_for_low', 0.3)
            self.logger.info(f"成功率很低，提高变异率: {current_mutation_rate:.3f} -> {new_mutation_rate:.3f}")
        elif success_rate < mutation_medium:
            # 成功率中等偏低，使用中等变异率
            new_mutation_rate = mutation_rules.get('rate_for_medium', 0.2)
            self.logger.info(f"成功率中等偏低，调整变异率: {current_mutation_rate:.3f} -> {new_mutation_rate:.3f}")
        elif success_rate >= mutation_high:
            # 成功率较高，降低变异率以减少破坏性变化
            new_mutation_rate = mutation_rules.get('rate_for_high', 0.1)
            self.logger.info(f"成功率较高，降低变异率: {current_mutation_rate:.3f} -> {new_mutation_rate:.3f}")
        
        if abs(new_mutation_rate - current_mutation_rate) > 0.01:  # 避免微小变化
            self.config_manager.set_runtime_param("ga", "mutation_probability", new_mutation_rate)
            self.logger.info(f"变异率已调整为: {new_mutation_rate:.3f}")
        
        # 记录自适应调整总结
        self.logger.info(f"自适应调整完成 - 成功率: {success_rate:.4f}, 种群大小: {new_pop_size}, 交叉率: {new_crossover_rate:.3f}, 变异率: {new_mutation_rate:.3f}")
    
    def _save_checkpoint(self, population: List[Individual], generation: int) -> None:
        """
        保存当前优化进度的检查点。
        
        参数:
            population (List[Individual]): 当前种群
            generation (int): 当前代数
        """
        try:
            # 这里应该实现检查点保存逻辑
            # 可以保存到文件或通过result_handler保存
            self.logger.debug(f"保存检查点: 第 {generation} 代，种群大小 {len(population)}")
        except Exception as e:
            self.logger.error(f"保存检查点时发生错误: {e}")
    
    def _load_population_from_checkpoint(self) -> Optional[List[Individual]]:
        """
        从检查点加载种群。
        
        返回:
            Optional[List[Individual]]: 加载的种群，如果失败则返回None
        """
        try:
            # 占位符实现 - 实际应该从文件或数据库加载
            self.logger.debug("尝试从检查点加载种群")
            return None  # 暂时返回None，表示没有可用的检查点
        except Exception as e:
            self.logger.error(f"从检查点加载种群时发生错误: {e}")
            return None
    
    def _load_generation_from_checkpoint(self) -> int:
        """
        从检查点加载当前代数。
        
        返回:
            int: 当前代数
        """
        # 占位符实现
        return 0
    
    def _save_successful_alphas(self, successful_alphas: List[Individual]) -> None:
        """
        保存成功的Alpha个体。
        
        参数:
            successful_alphas (List[Individual]): 成功的Alpha个体列表
        """
        try:
            # 这里应该通过result_handler保存成功的Alpha
            self.logger.info(f"保存 {len(successful_alphas)} 个成功Alpha")
            for i, alpha in enumerate(successful_alphas):
                self.logger.info(f"成功Alpha {i+1}: {alpha.get_expression_str()}, 适应度: {alpha.fitness_score:.4f}")
        except Exception as e:
            self.logger.error(f"保存成功Alpha时发生错误: {e}")
    
    def _cleanup_checkpoints(self) -> None:
        """
        清理检查点文件。
        """
        try:
            # 占位符实现 - 实际应该删除临时检查点文件
            self.logger.debug("清理检查点文件")
        except Exception as e:
            self.logger.error(f"清理检查点文件时发生错误: {e}")


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

