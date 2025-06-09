# -*- coding: utf-8 -*-
import logging
import os
import json
import pandas as pd
import math # 需要用于 isnan, isclose, isfinite
from datetime import datetime # 需要用于时间戳
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple

# 项目内部模块导入
if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .wqb_api_client import WQB_API_Client
    from .alpha_evolution_engine import AlphaEvolutionEngine, Individual # Individual也可能需要
    # from .utils import Node # 如果 Node 类被直接引用的话

class ResultHandler:
    """
    结果处理器模块。
    （文档字符串与之前相同）
    """
    def __init__(self,
                 config_manager: 'ConfigManager',
                 wqb_api_client: 'WQB_API_Client',
                 alpha_evolution_engine_ref: Optional['AlphaEvolutionEngine'] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResultHandler 初始化开始...")

        self.config_manager: 'ConfigManager' = config_manager
        self.wqb_api_client: 'WQB_API_Client' = wqb_api_client
        # 注意: alpha_evolution_engine_ref 已在需求中更名为 alpha_evolution_engine
        # 我将使用 alpha_evolution_engine 作为属性名以匹配后续需求
        self.alpha_evolution_engine: Optional['AlphaEvolutionEngine'] = alpha_evolution_engine_ref

        data_base_path = self.config_manager.get("general.data_paths.base_data_directory", "data")
        self.successful_alphas_dir: str = os.path.join(
            data_base_path,
            self.config_manager.get("general.data_paths.successful_alphas_subdir", "successful_alphas")
        )
        self.alpha_pnl_dir: str = os.path.join(
            data_base_path,
            self.config_manager.get("general.data_paths.alpha_pnl_subdir", "alpha_pnl")
        )
        self.alpha_yearly_stats_dir: str = os.path.join(
            data_base_path,
            self.config_manager.get("general.data_paths.alpha_yearly_stats_subdir", "alpha_yearly_stats")
        )
        self.ga_checkpoints_dir: str = os.path.join(
            data_base_path,
            self.config_manager.get("general.data_paths.ga_checkpoints_subdir", "ga_checkpoints")
        )
        self._ensure_directories_exist()
        self.logger.info("ResultHandler 初始化完毕。")
        self.logger.debug(f"  合格Alpha保存路径: {self.successful_alphas_dir}")
        # ... 其他路径日志

    def _ensure_directories_exist(self) -> None:
        directories_to_check = [
            self.successful_alphas_dir, self.alpha_pnl_dir,
            self.alpha_yearly_stats_dir, self.ga_checkpoints_dir
        ]
        for directory in directories_to_check:
            try:
                os.makedirs(directory, exist_ok=True)
                self.logger.debug(f"目录已确认/创建: {directory}")
            except OSError as e:
                self.logger.error(f"创建目录 {directory} 失败: {e}", exc_info=True)
                raise RuntimeError(f"无法创建必要的数据存储目录: {directory}, Error: {e}") from e

    def set_alpha_evolution_engine(self, engine: 'AlphaEvolutionEngine') -> None:
        if self.alpha_evolution_engine is not None: # 统一使用 self.alpha_evolution_engine
            self.logger.warning(
                "AlphaEvolutionEngine实例已被设置过，将被新的实例覆盖。"
                "旧引用 ID: %s, 新引用 ID: %s",
                id(self.alpha_evolution_engine), id(engine)
            )
        self.alpha_evolution_engine = engine
        self.logger.info(f"AlphaEvolutionEngine实例已成功注入到ResultHandler。Engine ID: {id(engine)}")
        if self.alpha_evolution_engine is None:
             self.logger.critical("AlphaEvolutionEngine实例注入失败或仍为None，部分功能可能无法正常工作！")

    def handle_simulation_result(self,
                               alpha_id: str,
                               original_payload: Dict[str, Any],
                               wqb_raw_result: Dict[str, Any]
                               ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        处理单个Alpha的WQB模拟结果。

        此方法负责从原始API响应中提取性能指标，根据配置获取并保存额外数据（如PnL、年度统计），
        调用AlphaEvolutionEngine计算适应度，判断Alpha是否合格，并在合格时触发详细数据的保存。

        参数:
            alpha_id (str): WQB平台返回的此Alpha的唯一标识符。
            original_payload (Dict[str, Any]): 最初提交给WQB API进行模拟的请求payload。
            wqb_raw_result (Dict[str, Any]): WQB API返回的原始模拟结果JSON反序列化后的字典。

        返回:
            Tuple[bool, float, Dict[str, Any]]: 一个元组，包含：
                - is_qualified (bool): Alpha是否合格。
                - fitness_score (float): 计算得到的适应度分数。
                - processed_data (Dict[str, Any]): 包含所有提取、生成和保存的Alpha相关信息的字典。
        """
        self.logger.info(f"开始处理Alpha ID: {alpha_id} 的模拟结果。")

        if not alpha_id or not wqb_raw_result:
            self.logger.warning(f"接收到空或无效的WQB原始结果。Alpha ID: {alpha_id}. 结果: {wqb_raw_result}")
            return False, -float('inf'), {"error": "无效的原始结果或Alpha ID", "alpha_id": alpha_id}

        # 注意：适应度计算现在在 ResultHandler 内部实现，不再依赖 AlphaEvolutionEngine

        processed_data: Dict[str, Any] = {
            "alpha_id": alpha_id,
            "expression_str": original_payload.get('regular', "未知表达式"),
            "simulation_settings": original_payload.get('settings', {}),
            "wqb_raw_result": wqb_raw_result,
            "processing_timestamp": datetime.now().isoformat()
        }

        is_stats = wqb_raw_result.get('is', {})
        if not isinstance(is_stats, dict):
            self.logger.warning(f"Alpha {alpha_id}: WQB结果中的 'is' 字段不是预期的字典类型或不存在。Stats: {is_stats}")
            is_stats = {}

        # 提取指标，处理None和NaN
        def _get_metric(metric_name: str, default_value: float = 0.0) -> float:
            val = is_stats.get(metric_name)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                self.logger.debug(f"Alpha {alpha_id}: 指标 '{metric_name}' 为None或NaN，使用默认值 {default_value}。")
                return default_value
            try:
                return float(val)
            except (ValueError, TypeError):
                self.logger.warning(f"Alpha {alpha_id}: 指标 '{metric_name}' 的值 '{val}' 无法转换为float，使用默认值 {default_value}。")
                return default_value

        processed_data['sharpe'] = _get_metric('sharpe')
        processed_data['fitness'] = _get_metric('fitness') # WQB platform fitness
        processed_data['returns'] = _get_metric('returns')
        processed_data['drawdown'] = _get_metric('drawdown')
        processed_data['turnover'] = _get_metric('turnover')
        # ... 其他指标

        wqb_checks = is_stats.get('checks', [])
        if isinstance(wqb_checks, list):
            processed_data['wq_is_valid'] = all(
                check.get('result', '').upper() != 'FAIL' for check in wqb_checks if isinstance(check, dict)
            )
        else:
            self.logger.warning(f"Alpha {alpha_id}: WQB结果中的 'is.checks' 字段不是预期的列表类型。Checks: {wqb_checks}")
            processed_data['wq_is_valid'] = False

        self.logger.debug(f"Alpha {alpha_id}: 提取的IS指标 - Sharpe={processed_data['sharpe']:.4f}, WQFitness={processed_data['fitness']:.4f}, WQValid={processed_data['wq_is_valid']}")

        try:
            if self.config_manager.get("general.get_pnl", False):
                self.logger.debug(f"Alpha {alpha_id}: 配置要求获取PnL数据。")
                pnl_df = self.wqb_api_client.get_alpha_pnl(alpha_id, wqb_raw_result) # type: ignore # Mocked method
                if pnl_df is not None and not pnl_df.empty:
                    pnl_filename = f"{alpha_id}_pnl.csv"
                    pnl_file_path = self._save_df_to_csv(pnl_df, self.alpha_pnl_dir, pnl_filename, index=False)
                    if pnl_file_path: processed_data['pnl_file_path'] = pnl_file_path
                elif pnl_df is None: self.logger.warning(f"Alpha {alpha_id}: 获取PnL数据失败。")
                else: self.logger.info(f"Alpha {alpha_id}: PnL数据为空。")
        except Exception as e_pnl:
            self.logger.error(f"Alpha {alpha_id}: 获取或保存PnL数据时发生错误: {e_pnl}", exc_info=True)

        try:
            if self.config_manager.get("general.get_yearly_stats", False):
                self.logger.debug(f"Alpha {alpha_id}: 配置要求获取年度统计数据。")
                yearly_stats_df = self.wqb_api_client.get_alpha_yearly_stats(alpha_id, wqb_raw_result) # type: ignore # Mocked method
                if yearly_stats_df is not None and not yearly_stats_df.empty:
                    yearly_filename = f"{alpha_id}_yearly_stats.csv"
                    yearly_file_path = self._save_df_to_csv(yearly_stats_df, self.alpha_yearly_stats_dir, yearly_filename, index=False)
                    if yearly_file_path: processed_data['yearly_stats_file_path'] = yearly_file_path
                elif yearly_stats_df is None: self.logger.warning(f"Alpha {alpha_id}: 获取年度统计数据失败。")
                else: self.logger.info(f"Alpha {alpha_id}: 年度统计数据为空。")
        except Exception as e_yearly:
            self.logger.error(f"Alpha {alpha_id}: 获取或保存年度统计数据时发生错误: {e_yearly}", exc_info=True)

        # 计算适应度分数
        fitness_score: float = self._calculate_fitness(processed_data)
        processed_data['fitness_score_calculated'] = fitness_score
        self.logger.info(f"Alpha {alpha_id}: 计算得到的适应度分数为: {fitness_score:.6f}")

        min_acceptable_fitness = self.config_manager.get("ga.fitness_function.min_acceptable_fitness", 0.5)
        is_qualified: bool = (
            processed_data.get('wq_is_valid', False) and
            fitness_score >= min_acceptable_fitness and
            math.isfinite(fitness_score)
        )
        processed_data['is_qualified_calculated'] = is_qualified
        self.logger.info(f"Alpha {alpha_id}: 合格性判断 - WQValid={processed_data.get('wq_is_valid')}, Fitness={fitness_score:.4f}, MinAcceptable={min_acceptable_fitness}, Qualified={is_qualified}")

        if is_qualified:
            try:
                details_file_path = self._save_qualified_alpha_details(processed_data)
                if details_file_path:
                    processed_data['details_file_path'] = details_file_path
                else:
                    self.logger.error(f"Alpha {alpha_id}: 合格Alpha判定成功，但详细数据保存失败！")
            except Exception as e_save_details:
                self.logger.error(f"Alpha {alpha_id}: 保存合格Alpha详细数据时发生错误: {e_save_details}", exc_info=True)

        self.logger.info(f"Alpha ID: {alpha_id} 处理完毕。最终合格状态: {is_qualified}, 适应度: {fitness_score:.6f}")
        return is_qualified, fitness_score, processed_data

    def _save_df_to_csv(self, df: pd.DataFrame, directory: str, filename: str, index: bool = False, **kwargs) -> Optional[str]:
        """
        辅助方法：将DataFrame保存为CSV文件。
        (占位符 - 将在T5.1中实现)
        """
        self.logger.debug(f"占位符: _save_df_to_csv 调用，目标 {os.path.join(directory, filename)}")
        # 实际实现:
        # if not os.path.exists(directory): os.makedirs(directory, exist_ok=True)
        # file_path = os.path.join(directory, filename)
        # try:
        #     df.to_csv(file_path, index=index, **kwargs)
        #     return file_path
        # except Exception as e:
        #     self.logger.error(f"保存DataFrame到CSV '{file_path}' 失败: {e}", exc_info=True)
        #     return None
        pass # 占位符
        return None # 占位符返回

    def _calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """
        计算Alpha的适应度分数。
        
        使用ConfigManager中的配置权重和阈值，计算复合适应度分数。
        支持多种指标的加权组合，包括Sharpe比率、WQB平台适应度、收益率、回撤等。
        
        参数:
            metrics (Dict[str, Any]): 包含各种性能指标的字典
            
        返回:
            float: 计算得到的适应度分数，如果计算失败返回负无穷
        """
        try:
            # 从配置中获取适应度函数设置
            fitness_config = self.config_manager.get("ga.fitness_function", {})
            
            # 提取各项指标，处理None和NaN值
            sharpe = metrics.get('sharpe', 0.0)
            wq_fitness = metrics.get('fitness', 0.0)  # WQB平台的适应度
            returns = metrics.get('returns', 0.0)
            drawdown = metrics.get('drawdown', 0.0)
            turnover = metrics.get('turnover', 0.0)
            
            # 确保所有指标都是有效的数值
            def _safe_float(value: Any, default: float = 0.0) -> float:
                if value is None:
                    return default
                try:
                    val = float(value)
                    return val if math.isfinite(val) else default
                except (ValueError, TypeError):
                    return default
            
            sharpe = _safe_float(sharpe)
            wq_fitness = _safe_float(wq_fitness)
            returns = _safe_float(returns)
            drawdown = _safe_float(drawdown)
            turnover = _safe_float(turnover)
            
            # 从配置中获取各指标的权重
            weight_sharpe = fitness_config.get('weight_sharpe', 0.4)
            weight_wq_fitness = fitness_config.get('weight_wq_fitness', 0.3)
            weight_returns = fitness_config.get('weight_returns', 0.2)
            weight_drawdown = fitness_config.get('weight_drawdown', -0.1)  # 负权重，回撤越小越好
            weight_turnover = fitness_config.get('weight_turnover', -0.05)  # 负权重，换手率越小越好
            
            # 计算加权适应度分数
            fitness_score = (
                sharpe * weight_sharpe +
                wq_fitness * weight_wq_fitness +
                returns * weight_returns +
                drawdown * weight_drawdown +  # 回撤为负值时，负权重使其贡献为正
                turnover * weight_turnover
            )
            
            # 应用额外的惩罚或奖励机制
            # 如果Sharpe比率过低，给予额外惩罚
            min_sharpe_threshold = fitness_config.get('min_sharpe_threshold', 0.5)
            if sharpe < min_sharpe_threshold:
                sharpe_penalty = fitness_config.get('sharpe_penalty', 0.2)
                fitness_score -= sharpe_penalty
                self.logger.debug(f"Sharpe比率 {sharpe:.4f} 低于阈值 {min_sharpe_threshold}，应用惩罚 {sharpe_penalty}")
            
            # 如果回撤过大，给予额外惩罚
            max_drawdown_threshold = fitness_config.get('max_drawdown_threshold', -0.2)
            if drawdown < max_drawdown_threshold:  # drawdown通常为负值
                drawdown_penalty = fitness_config.get('drawdown_penalty', 0.3)
                fitness_score -= drawdown_penalty
                self.logger.debug(f"回撤 {drawdown:.4f} 超过阈值 {max_drawdown_threshold}，应用惩罚 {drawdown_penalty}")
            
            # 确保最终分数是有限的数值
            if not math.isfinite(fitness_score):
                self.logger.warning(f"计算得到的适应度分数不是有限数值: {fitness_score}，返回负无穷")
                return -float('inf')
            
            self.logger.debug(
                f"适应度计算详情: sharpe={sharpe:.4f}*{weight_sharpe} + "
                f"wq_fitness={wq_fitness:.4f}*{weight_wq_fitness} + "
                f"returns={returns:.4f}*{weight_returns} + "
                f"drawdown={drawdown:.4f}*{weight_drawdown} + "
                f"turnover={turnover:.4f}*{weight_turnover} = {fitness_score:.6f}"
            )
            
            return fitness_score
            
        except Exception as e:
            self.logger.error(f"计算适应度分数时发生错误: {e}", exc_info=True)
            return -float('inf')

    def _save_qualified_alpha_details(self, alpha_data: Dict[str, Any]) -> Optional[str]:
        """
        辅助方法：将合格Alpha的详细数据保存为JSON文件。
        (占位符 - 将在T5.1中实现)
        """
        alpha_id = alpha_data.get("alpha_id", "unknown_alpha")
        filename = f"{alpha_id}_details.json"
        self.logger.debug(f"占位符: _save_qualified_alpha_details 调用，目标 {os.path.join(self.successful_alphas_dir, filename)}")
        # 实际实现:
        # file_path = os.path.join(self.successful_alphas_dir, filename)
        # try:
        #     with open(file_path, 'w', encoding='utf-8') as f:
        #         json.dump(alpha_data, f, indent=4, ensure_ascii=False)
        #     return file_path
        # except Exception as e:
        #     self.logger.error(f"保存合格Alpha详情到JSON '{file_path}' 失败: {e}", exc_info=True)
        #     return None
        pass # 占位符
        return None # 占位符返回

    def save_ga_state(self, population: list, generation: int, total_successful: int) -> bool:
        """
        保存遗传算法的当前状态到检查点文件。
        
        此方法将当前GA状态（种群、代数、合格Alpha总数）序列化并保存到JSON文件中。
        Individual实例中的AST将转换为字符串格式进行存储。使用原子性写入防止文件损坏。
        
        参数:
            population (list): 包含Individual实例的当前种群列表。
            generation (int): 当前遗传算法代数。
            total_successful (int): 已发现的合格Alpha总数。
            
        返回:
            bool: 保存成功返回True，失败返回False。
        """
        self.logger.info(f"开始保存GA状态: 代数={generation}, 种群大小={len(population)}, 合格Alpha总数={total_successful}")
        
        try:
            # 导入必要的模块
            import json
            import tempfile
            from src.utils import generate_expression_from_ast
            
            # 构建检查点文件名
            checkpoint_filename = f"ga_checkpoint_{generation:04d}.json"
            checkpoint_path = os.path.join(self.ga_checkpoints_dir, checkpoint_filename)
            
            # 序列化种群数据
            serialized_population = []
            for individual in population:
                try:
                    # 将Individual的AST转换为字符串表达式
                    expression_str = generate_expression_from_ast(individual.expression_ast)
                    
                    # 构建序列化的Individual数据
                    serialized_individual = {
                        'expression_str': expression_str,  # AST转换为字符串
                        'simulation_params': individual.simulation_params,
                        'fitness_score': individual.fitness_score,
                        'alpha_id': individual.alpha_id
                    }
                    serialized_population.append(serialized_individual)
                    
                except Exception as e:
                    self.logger.error(f"序列化Individual时发生错误 (alpha_id: {getattr(individual, 'alpha_id', 'unknown')}): {e}")
                    # 继续处理其他Individual，不中断整个保存过程
                    continue
            
            # 构建完整的GA状态数据
            ga_state_data = {
                'generation': generation,
                'total_successful_alphas': total_successful,
                'population_size': len(serialized_population),
                'population': serialized_population,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'  # 用于未来兼容性
            }
            
            # 使用原子性写入：先写入临时文件，然后重命名
            temp_fd, temp_path = tempfile.mkstemp(dir=self.ga_checkpoints_dir, suffix='.tmp')
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                    json.dump(ga_state_data, temp_file, indent=2, ensure_ascii=False)
                
                # 原子性重命名操作
                if os.path.exists(checkpoint_path):
                    # 备份现有文件
                    backup_path = checkpoint_path + '.backup'
                    os.rename(checkpoint_path, backup_path)
                    self.logger.debug(f"已备份现有检查点文件: {backup_path}")
                
                os.rename(temp_path, checkpoint_path)
                self.logger.info(f"GA状态已成功保存到: {checkpoint_path}")
                
                # 清理备份文件（可选）
                backup_path = checkpoint_path + '.backup'
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    
                return True
                
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
                
        except Exception as e:
            self.logger.error(f"保存GA状态时发生错误: {e}", exc_info=True)
            return False
    
    def load_ga_state(self) -> tuple[list, int, int]:
        """
        从检查点文件加载遗传算法状态。
        
        此方法查找最新的GA检查点文件，反序列化JSON内容，并重新构建Individual实例。
        字符串表达式将转换回AST结构。
        
        返回:
            tuple[list, int, int]: 包含以下元素的元组：
                - population (list): 重新构建的Individual实例列表
                - generation (int): 加载的代数
                - total_successful (int): 加载的合格Alpha总数
            如果找不到检查点文件或加载失败，返回 ([], 0, 0)
        """
        self.logger.info("开始加载GA状态...")
        
        try:
            import json
            import glob
            from src.utils import parse_expression_to_ast
            
            # 查找最新的检查点文件
            checkpoint_pattern = os.path.join(self.ga_checkpoints_dir, "ga_checkpoint_*.json")
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if not checkpoint_files:
                self.logger.info("未找到GA检查点文件，返回初始状态")
                return [], 0, 0
            
            # 按文件名排序，获取最新的检查点
            checkpoint_files.sort()
            latest_checkpoint = checkpoint_files[-1]
            self.logger.info(f"找到最新检查点文件: {latest_checkpoint}")
            
            # 加载检查点数据
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                ga_state_data = json.load(f)
            
            # 提取基本状态信息
            generation = ga_state_data.get('generation', 0)
            total_successful = ga_state_data.get('total_successful_alphas', 0)
            population_data = ga_state_data.get('population', [])
            
            self.logger.info(f"加载GA状态: 代数={generation}, 合格Alpha总数={total_successful}, 种群大小={len(population_data)}")
            
            # 重新构建Individual实例
            # 注意：这里需要导入Individual类，但由于循环导入问题，我们延迟导入
            try:
                from src.alpha_evolution_engine import Individual
            except ImportError as ie:
                self.logger.error(f"无法导入Individual类: {ie}")
                return [], 0, 0
            
            reconstructed_population = []
            for individual_data in population_data:
                try:
                    # 将字符串表达式转换回AST
                    expression_str = individual_data.get('expression_str', '')
                    expression_ast = parse_expression_to_ast(expression_str)
                    
                    if expression_ast is None:
                        self.logger.warning(f"无法解析表达式: {expression_str}，跳过此Individual")
                        continue
                    
                    # 重新构建Individual实例
                    individual = Individual(
                        expression_ast=expression_ast,
                        simulation_params=individual_data.get('simulation_params', {}),
                        fitness_score=individual_data.get('fitness_score', 0.0),
                        alpha_id=individual_data.get('alpha_id', '')
                    )
                    
                    reconstructed_population.append(individual)
                    
                except Exception as e:
                    self.logger.error(f"重构Individual时发生错误: {e}")
                    # 继续处理其他Individual
                    continue
            
            self.logger.info(f"成功重构 {len(reconstructed_population)} 个Individual实例")
            return reconstructed_population, generation, total_successful
            
        except FileNotFoundError:
            self.logger.info("检查点文件不存在，返回初始状态")
            return [], 0, 0
        except json.JSONDecodeError as je:
            self.logger.error(f"检查点文件JSON格式错误: {je}")
            return [], 0, 0
        except Exception as e:
            self.logger.error(f"加载GA状态时发生错误: {e}", exc_info=True)
            return [], 0, 0

# --- 用于测试的模拟类和 if __name__ == '__main__' 代码块 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger(__name__ + "._rh_test_")

    # --- 模拟依赖类 ---
    class MockConfigManager:
        def __init__(self):
            self._config = {
                "data_paths.base_data_directory": "test_data_rh_t5",
                "data_paths.successful_alphas_subdir": "good_alphas",
                "data_paths.alpha_pnl_subdir": "pnl_data",
                "data_paths.alpha_yearly_stats_subdir": "yearly_data",
                "data_paths.ga_checkpoints_subdir": "ga_saves",
                "general.get_pnl": True,
                "general.get_yearly_stats": True,
                "ga.fitness_function.min_acceptable_fitness": 0.7,
                "ga.fitness_function.weight_sharpe": 0.5,
                "ga.fitness_function.weight_wq_fitness": 0.3,
                "ga.fitness_function.weight_returns": 0.1,
                "ga.fitness_function.weight_drawdown": 0.05, # 确保为正，计算时用abs(drawdown)
                "ga.fitness_function.weight_turnover": 0.05,
                "ga.fitness_function.turnover_penalty_factor": 0.9,
                "ga.fitness_function.max_turnover_threshold": 2.5,
                "ga.fitness_function.min_turnover_threshold": 0.1, # 新增
                "ga.fitness_function.invalid_penalty_factor": 0.1,
                "ga.fitness_function.epsilon": 1e-6,
                "ga.fitness_function.turnover_power": 1.0 # 新增
            }
            base_test_dir = self._config["data_paths.base_data_directory"]
            if os.path.exists(base_test_dir):
                import shutil
                test_logger.debug(f"清理已存在的测试目录: {base_test_dir}")
                shutil.rmtree(base_test_dir)

        def get(self, key_path: str, default: Any = None) -> Any:
            return self._config.get(key_path, default)

    class MockWQBAPIClient:
        def get_alpha_pnl(self, alpha_id: str) -> Optional[pd.DataFrame]:
            test_logger.debug(f"MockWQBAPIClient.get_alpha_pnl called for {alpha_id}")
            if alpha_id == "alpha_error_pnl": return None
            if alpha_id == "alpha_empty_pnl": return pd.DataFrame()
            return pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'pnl': [0.1, -0.05]})

        def get_alpha_yearly_stats(self, alpha_id: str) -> Optional[pd.DataFrame]:
            test_logger.debug(f"MockWQBAPIClient.get_alpha_yearly_stats called for {alpha_id}")
            if alpha_id == "alpha_error_yearly": return None
            if alpha_id == "alpha_empty_yearly": return pd.DataFrame()
            return pd.DataFrame({'year': [2023], 'sharpe': [1.5], 'returns': [0.15]})

    if not TYPE_CHECKING: # Runtime placeholder for AlphaEvolutionEngine
        class AlphaEvolutionEngine:
            def __init__(self, config_manager): # Simplified mock
                self.config_manager = config_manager
                self.logger = logging.getLogger(__name__ + ".MockAEE")

            # 注意：calculate_fitness 方法已移除，因为适应度计算现在在 ResultHandler 中实现
            pass


    base_test_dir_main = "test_data_rh_t5"
    try:
        test_logger.info("开始测试 ResultHandler.handle_simulation_result...")
        mock_config_instance = MockConfigManager()
        mock_wqb_client_instance = MockWQBAPIClient()
        mock_engine_instance = AlphaEvolutionEngine(mock_config_instance) # type: ignore

        rh_instance = ResultHandler(
            config_manager=mock_config_instance, # type: ignore
            wqb_api_client=mock_wqb_client_instance, # type: ignore
            alpha_evolution_engine_ref=mock_engine_instance # type: ignore
        )

        # 测试用例1: 合格的Alpha
        payload1 = {"regular": "ts_rank(close, 10)", "settings": {"universe": "TOP1000"}}
        wqb_result1 = {
            "id": "alpha001",
            "is": {
                "sharpe": 1.5, "fitness": 0.8, "returns": 0.12,
                "drawdown": -0.05, "turnover": 0.8,
                "checks": [{"name": "check1", "result": "PASS"}]
            }
        }
        is_q1, fit1, proc1 = rh_instance.handle_simulation_result("alpha001", payload1, wqb_result1)
        test_logger.info(f"测试1结果: Qualified={is_q1}, Fitness={fit1:.4f}, DataKeys={list(proc1.keys())}")
        assert is_q1 is True
        assert proc1.get('pnl_file_path') is None # 因为存根返回None
        assert proc1.get('yearly_stats_file_path') is None # 因为存根返回None
        assert proc1.get('details_file_path') is None # 因为存根返回None
        assert 'fitness_score_calculated' in proc1

        # 测试用例2: WQB检查失败
        wqb_result2 = {
            "id": "alpha002",
            "is": {
                "sharpe": 1.6, "fitness": 0.9, "returns": 0.15,
                "drawdown": -0.04, "turnover": 0.7,
                "checks": [{"name": "check1", "result": "FAIL"}] # WQB check failed
            }
        }
        is_q2, fit2, proc2 = rh_instance.handle_simulation_result("alpha002", payload1, wqb_result2)
        test_logger.info(f"测试2结果 (WQ check fail): Qualified={is_q2}, Fitness={fit2:.4f}")
        assert is_q2 is False
        assert proc2.get('wq_is_valid') is False

        # 测试用例3: 适应度低于阈值
        wqb_result3 = {
            "id": "alpha003",
            "is": {
                "sharpe": 0.1, "fitness": 0.05, "returns": 0.01,
                "drawdown": -0.1, "turnover": 1.0,
                "checks": [{"name": "check1", "result": "PASS"}]
            }
        }
        is_q3, fit3, proc3 = rh_instance.handle_simulation_result("alpha003", payload1, wqb_result3)
        test_logger.info(f"测试3结果 (low fitness): Qualified={is_q3}, Fitness={fit3:.4f}")
        assert is_q3 is False

        # 测试用例4: 指标为None或NaN
        wqb_result4 = {
            "id": "alpha004",
            "is": {
                "sharpe": None, "fitness": math.nan, "returns": 0.12,
                "drawdown": -0.05, "turnover": 0.8,
                "checks": [{"name": "check1", "result": "PASS"}]
            }
        }
        is_q4, fit4, proc4 = rh_instance.handle_simulation_result("alpha004", payload1, wqb_result4)
        test_logger.info(f"测试4结果 (None/NaN metrics): Qualified={is_q4}, Fitness={fit4:.4f}")
        assert proc4['sharpe'] == 0.0
        assert proc4['fitness'] == 0.0
        # 适应度可能为0或负，取决于权重，但应是有限数

        # 测试用例5: 空的WQB结果
        test_logger.info("测试空的WQB结果...")
        is_q5, fit5, proc5 = rh_instance.handle_simulation_result("alpha005", payload1, {})
        test_logger.info(f"测试5结果 (empty WQB result): Qualified={is_q5}, Fitness={fit5}")
        assert is_q5 is False
        assert fit5 == -float('inf')
        assert 'error' in proc5

        test_logger.info("ResultHandler.handle_simulation_result 测试完成。")

    except Exception as e:
        test_logger.error(f"测试 ResultHandler.handle_simulation_result 时发生错误: {e}", exc_info=True)
    finally:
        if os.path.exists(base_test_dir_main):
            import shutil
            shutil.rmtree(base_test_dir_main)
            test_logger.info(f"测试目录 '{base_test_dir_main}' 已清理。")


