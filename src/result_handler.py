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

        if self.alpha_evolution_engine is None:
            self.logger.critical(
                "AlphaEvolutionEngine引用未在ResultHandler中设置！无法计算适应度。"
                "请确保在SystemOrchestrator中正确调用set_alpha_evolution_engine。"
            )
            raise RuntimeError("AlphaEvolutionEngine未设置，无法处理模拟结果。")

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
                pnl_df = self.wqb_api_client.get_alpha_pnl(alpha_id) # type: ignore # Mocked method
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
                yearly_stats_df = self.wqb_api_client.get_alpha_yearly_stats(alpha_id) # type: ignore # Mocked method
                if yearly_stats_df is not None and not yearly_stats_df.empty:
                    yearly_filename = f"{alpha_id}_yearly_stats.csv"
                    yearly_file_path = self._save_df_to_csv(yearly_stats_df, self.alpha_yearly_stats_dir, yearly_filename, index=False)
                    if yearly_file_path: processed_data['yearly_stats_file_path'] = yearly_file_path
                elif yearly_stats_df is None: self.logger.warning(f"Alpha {alpha_id}: 获取年度统计数据失败。")
                else: self.logger.info(f"Alpha {alpha_id}: 年度统计数据为空。")
        except Exception as e_yearly:
            self.logger.error(f"Alpha {alpha_id}: 获取或保存年度统计数据时发生错误: {e_yearly}", exc_info=True)

        fitness_score: float = -float('inf')
        try:
            fitness_score = self.alpha_evolution_engine.calculate_fitness(processed_data) # type: ignore # Mocked method
            processed_data['fitness_score_calculated'] = fitness_score
            self.logger.info(f"Alpha {alpha_id}: 计算得到的适应度分数为: {fitness_score:.6f}")
        except Exception as e_fitness:
            self.logger.error(f"Alpha {alpha_id}: 调用 calculate_fitness 时发生错误: {e_fitness}", exc_info=True)

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

            def calculate_fitness(self, metrics: Dict[str, Any]) -> float:
                # 模拟的适应度计算，实际应更复杂
                self.logger.debug(f"MockAEE.calculate_fitness called with metrics: {metrics}")
                fitness_config = self.config_manager.get("ga.fitness_function", {})
                sharpe = metrics.get('sharpe', 0.0)
                wq_fitness = metrics.get('fitness', 0.0)
                # 简单加权，真实逻辑在 AlphaEvolutionEngine.calculate_fitness
                fit = ( (sharpe if sharpe is not None else 0.0) * fitness_config.get('weight_sharpe', 0.5) +
                        (wq_fitness if wq_fitness is not None else 0.0) * fitness_config.get('weight_wq_fitness', 0.3) )
                return fit if math.isfinite(fit) else -float('inf')


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

        # 测试用例5: 引擎未设置
        rh_instance.alpha_evolution_engine = None # 模拟引擎未设置
        test_logger.info("测试引擎未设置 (预期RuntimeError)...")
        try:
            rh_instance.handle_simulation_result("alpha005", payload1, wqb_result1)
        except RuntimeError as re:
            test_logger.info(f"成功捕获到预期的RuntimeError: {re}")
        rh_instance.alpha_evolution_engine = mock_engine_instance # 恢复

        test_logger.info("ResultHandler.handle_simulation_result 测试完成。")

    except Exception as e:
        test_logger.error(f"测试 ResultHandler.handle_simulation_result 时发生错误: {e}", exc_info=True)
    finally:
        if os.path.exists(base_test_dir_main):
            import shutil
            shutil.rmtree(base_test_dir_main)
            test_logger.info(f"测试目录 '{base_test_dir_main}' 已清理。")


