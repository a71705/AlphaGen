# -*- coding: utf-8 -*-
import logging
import sys
import traceback # 后续用于打印完整异常堆栈

# --- 导入本项目核心模块的类 ---
# 使用 try-except ImportError 结构处理不同导入上下文
try:
    from .config_manager import ConfigManager
    from .wqb_api_client import WQB_API_Client
    from .alpha_evolution_engine import AlphaEvolutionEngine
    from .result_handler import ResultHandler
    from .utils import setup_logging # 假设 setup_logging 在 utils.py 中
except ImportError as e:
    print(f"SystemOrchestrator: Relative import failed ('{e}'), attempting absolute import fallback.", file=sys.stderr)
    try:
        from config_manager import ConfigManager
        from wqb_api_client import WQB_API_Client
        from alpha_evolution_engine import AlphaEvolutionEngine
        from result_handler import ResultHandler
        from utils import setup_logging
    except ImportError as e_abs:
        print(f"CRITICAL: SystemOrchestrator - Absolute import failed for core modules: {e_abs}. System cannot start.", file=sys.stderr)
        raise


class SystemOrchestrator:
    """
    系统编排器。

    作为整个自动化Alpha因子挖掘系统的总控制器和用户交互入口。
    负责初始化所有核心组件，处理它们之间的依赖关系，
    并提供一个主菜单驱动的界面来调用系统的各项功能。
    """

    def __init__(self):
        """
        初始化SystemOrchestrator。
        主要职责是调用私有方法 `_initialize_components` 来设置所有系统组件。
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("SystemOrchestrator 实例化开始...")

        self.config_manager: ConfigManager = None # type: ignore
        self.wqb_api_client: WQB_API_Client = None # type: ignore
        self.alpha_evolution_engine: AlphaEvolutionEngine = None # type: ignore
        self.result_handler: ResultHandler = None # type: ignore

        try:
            self._initialize_components()
            self.logger.info("SystemOrchestrator 核心组件初始化成功。")
        except Exception as e:
            critical_msg = f"SystemOrchestrator 初始化核心组件时发生致命错误: {e}"
            if hasattr(self, 'logger') and self.logger.handlers:
                self.logger.critical(critical_msg, exc_info=True)
            else:
                print(critical_msg, file=sys.stderr)
                traceback.print_exc()
            raise RuntimeError(f"系统核心组件初始化失败: {e}") from e

    def _initialize_components(self) -> None:
        """
        私有方法：初始化所有系统核心组件并处理它们之间的依赖关系。
        """
        try:
            setup_logging()
            self.logger = logging.getLogger(__name__)
            self.logger.info("全局日志系统已初始化。")
        except Exception as log_e:
            print(f"严重错误：日志系统初始化失败！{log_e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        self.logger.debug("正在实例化 ConfigManager...")
        try:
            self.config_manager = ConfigManager()
            self.logger.info("ConfigManager 已实例化。")
        except Exception as e:
            self.logger.critical(f"ConfigManager 实例化失败: {e}", exc_info=True)
            raise

        try:
            log_level_str = self.config_manager.get("general.log_level", "INFO").upper()
            numeric_log_level = getattr(logging, log_level_str, logging.INFO)
            logging.getLogger().setLevel(numeric_log_level)
            self.logger.info(f"全局日志级别已根据配置更新为: {log_level_str} ({numeric_log_level})")
        except Exception as e:
            self.logger.warning(f"根据配置更新日志级别失败: {e}. 将使用 setup_logging 设定的级别或默认级别。", exc_info=True)

        self.logger.debug("正在实例化 WQB_API_Client...")
        try:
            self.wqb_api_client = WQB_API_Client(config_manager=self.config_manager)
            self.logger.info("WQB_API_Client 已实例化。")
        except Exception as e:
            self.logger.critical(f"WQB_API_Client 实例化失败: {e}", exc_info=True)
            raise

        self.logger.debug("正在实例化 ResultHandler (alpha_evolution_engine 暂为 None)...")
        try:
            self.result_handler = ResultHandler(
                config_manager=self.config_manager,
                wqb_api_client=self.wqb_api_client,
                alpha_evolution_engine_ref=None
            )
            self.logger.info("ResultHandler 已实例化。")
        except Exception as e:
            self.logger.critical(f"ResultHandler 实例化失败: {e}", exc_info=True)
            raise

        self.logger.debug("正在实例化 AlphaEvolutionEngine...")
        try:
            self.alpha_evolution_engine = AlphaEvolutionEngine(
                config_manager=self.config_manager,
                wqb_api_client=self.wqb_api_client,
                result_handler=self.result_handler
            )
            self.logger.info("AlphaEvolutionEngine 已实例化。")
        except Exception as e:
            self.logger.critical(f"AlphaEvolutionEngine 实例化失败: {e}", exc_info=True)
            raise

        self.logger.debug("正在将 AlphaEvolutionEngine 实例注入到 ResultHandler...")
        try:
            self.result_handler.set_alpha_evolution_engine(self.alpha_evolution_engine)
            self.logger.info("AlphaEvolutionEngine 实例已成功注入 ResultHandler。")
        except Exception as e:
            self.logger.error(f"将 AlphaEvolutionEngine 注入 ResultHandler 时失败: {e}", exc_info=True)

        self.logger.info("所有核心系统组件均已成功初始化和配置。")

    # --- start() 方法 ---
    def start(self) -> None:
        """
        启动系统编排器的主要执行流程。
        此方法首先尝试确保与WQB API的会话有效（或执行登录），
        然后进入主菜单循环，等待用户指令。
        它还包含了顶层的异常处理，以确保系统的优雅退出。
        """
        self.logger.info("SystemOrchestrator 启动...")

        if not all([self.config_manager, self.wqb_api_client, self.alpha_evolution_engine, self.result_handler]):
            self.logger.critical("一个或多个核心组件未能初始化，系统无法启动。请检查初始化日志。")
            print("错误：系统核心组件初始化失败，无法启动。请检查日志。")
            sys.exit(1)

        try:
            self.logger.info("尝试刷新WQB API会话或执行登录...")
            self.wqb_api_client.refresh_session_if_needed()
            self.logger.info("WQB API会话已准备就绪或登录尝试完毕。")
        except FileNotFoundError as e_fnf:
            self.logger.critical(f"WQB凭证文件未找到: {e_fnf}. API相关功能将受限。", exc_info=False) # exc_info=False 简化日志
            print(f"\n警告：WQB凭证文件 ({self.wqb_api_client.credentials_file if self.wqb_api_client else '路径未知'}) 未找到。")
            print("部分需要API认证的功能可能无法使用。您可以稍后在菜单中尝试更新数据字段等操作（可能触发登录）。")
        except Exception as e_login:
            self.logger.error(f"WQB API会话准备过程中发生错误: {e_login}", exc_info=True)
            print(f"\n警告：WQB API会话准备失败 - {e_login}")
            print("需要API认证的功能可能无法正常工作。请检查网络连接、API配置和凭证。")

        try:
            self._main_menu_loop()
        except KeyboardInterrupt:
            self.logger.info("检测到用户中断 (Ctrl+C)，系统将优雅退出...")
            print("\n系统已接收到用户中断信号，正在退出...")
        except Exception as e_main:
            self.logger.critical(f"在主执行流程中发生未捕获的致命错误: {e_main}", exc_info=True)
            print(f"\n发生严重系统错误: {e_main}")
            traceback.print_exc()
        finally:
            self.logger.info("SystemOrchestrator 正在关闭。")
            # 尝试关闭已知的线程池
            if hasattr(self.wqb_api_client, '_batch_submit_executor') and self.wqb_api_client._batch_submit_executor:
                self.logger.debug("正在关闭 WQB API Client 的批处理提交执行器...")
                self.wqb_api_client._batch_submit_executor.shutdown(wait=False)
            if hasattr(self.wqb_api_client, '_batch_monitor_executor') and self.wqb_api_client._batch_monitor_executor:
                self.logger.debug("正在关闭 WQB API Client 的批处理监控执行器...")
                self.wqb_api_client._batch_monitor_executor.shutdown(wait=False)

            # AlphaEvolutionEngine 可能也有自己的线程池 (例如用于并发评估)
            # 假设其评估线程池名为 _evaluation_executor (如果存在)
            if hasattr(self.alpha_evolution_engine, '_evaluation_executor') and \
               self.alpha_evolution_engine._evaluation_executor: # type: ignore
                self.logger.debug("正在关闭 Alpha Evolution Engine 的评估执行器...")
                self.alpha_evolution_engine._evaluation_executor.shutdown(wait=False) # type: ignore
            self.logger.info("已尝试关闭已知的线程池。")

        print("感谢使用Alpha因子挖掘系统！")

    # --- _main_menu_loop() 方法 ---
    def _main_menu_loop(self) -> None:
        """
        显示主菜单并处理用户输入，循环执行直到用户选择退出。
        """
        while True:
            print("\n" + "="*40)
            print("  WorldQuant Alpha 因子自动化挖掘系统")
            print("="*40)
            print("  主菜单:")
            print("  1. 启动 Alpha 挖掘 (遗传优化)")
            print("  2. 爬取/更新 WQB 数据集和数据字段")
            print("  3. 查看已发现的合格 Alpha")
            print("  4. 系统设置")
            print("  5. 黑名单管理")
            print("  6. 退出系统")
            print("="*40)

            choice = input("请输入您的选择 (1-6): ").strip()
            self.logger.debug(f"用户菜单选择: '{choice}'")

            try:
                if choice == '1':
                    self._run_alpha_mining()
                elif choice == '2':
                    self._update_datafields()
                elif choice == '3':
                    self._view_successful_alphas()
                elif choice == '4':
                    self._system_settings()
                elif choice == '5':
                    self._manage_blacklist()
                elif choice == '6':
                    self.logger.info("用户选择退出系统。")
                    print("正在退出系统...")
                    break
                else:
                    self.logger.warning(f"无效的菜单选择: '{choice}'")
                    print(f"无效选择 '{choice}'。请输入1到6之间的数字。")

            except KeyboardInterrupt:
                self.logger.info("用户在执行菜单操作时按下Ctrl+C，返回主菜单。")
                print("\n操作被用户中断，正在返回主菜单...")
            except Exception as e_menu_item:
                self.logger.error(f"执行菜单选项 '{choice}' 时发生错误: {e_menu_item}", exc_info=True)
                print(f"\n执行操作 '{choice}' 失败: {e_menu_item}")
                print("请查看日志了解详细信息。正在返回主菜单。")

            if choice != '6':
                input("\n按 Enter键 返回主菜单...")

    # --- 各菜单功能处理方法的实现 (使用T6.2中已有的功能性实现) ---
    def _run_alpha_mining(self) -> None:
        """处理用户启动Alpha挖掘（遗传优化）的请求。"""
        self.logger.info("用户选择启动Alpha挖掘流程。")
        print("\n--- 启动 Alpha 因子挖掘 ---")

        resume_choice_str = input("是否从上次的进度恢复？(y/n，默认为 n): ").strip().lower()
        resume_ga = (resume_choice_str == 'y')
        self.logger.info(f"Alpha挖掘流程，是否恢复: {resume_ga}")

        try:
            print(f"正在{'恢复并' if resume_ga else '开始新的'}Alpha挖掘过程，这可能需要较长时间...")
            print("您可以随时按 Ctrl+C 来尝试中断挖掘过程（系统会尝试保存当前进度）。")

            if not self.alpha_evolution_engine:
                self.logger.error("AlphaEvolutionEngine 未初始化，无法启动挖掘。")
                print("错误：Alpha进化引擎未准备就绪，无法启动挖掘。请检查系统初始化日志。")
                return

            self.alpha_evolution_engine.run_optimization(resume=resume_ga) # 假设此方法已实现

            self.logger.info("Alpha挖掘流程正常完成。")
            print("\nAlpha挖掘流程已完成。")

        except KeyboardInterrupt:
            self.logger.warning("Alpha挖掘流程被用户通过Ctrl+C中断。")
            print("\nAlpha挖掘流程已被用户中断。系统（应已）尝试保存当前进度。")
        except Exception as e:
            self.logger.error(f"Alpha挖掘过程中发生未知严重错误: {e}", exc_info=True)
            print(f"\n在Alpha挖掘过程中发生严重错误: {e}")
            print("请检查日志获取详细信息。")

    def _update_datafields(self) -> None:
        """处理用户请求爬取或更新WQB可用数据集和数据字段列表。"""
        self.logger.info("用户选择更新WQB数据集和数据字段列表。")
        print("\n--- 更新 WQB 数据集和数据字段列表 ---")

        if not self.wqb_api_client:
            self.logger.error("WQBAPIClient 未初始化，无法更新数据字段。")
            print("错误：WQB API客户端未准备就绪。请检查系统初始化日志。")
            return

        try:
            # 第一步：获取数据集列表
            print("第一步：正在从WQB平台获取可用的数据集列表，请稍候...")
            datasets_list = self.wqb_api_client.get_available_datafields()  # 这个方法现在返回数据集列表

            if datasets_list is not None and len(datasets_list) > 0:
                datasets_count = len(datasets_list)
                self.logger.info(f"成功获取并更新了 {datasets_count} 个WQB数据集。")
                print(f"\n成功更新了 {datasets_count} 个数据集。")
                
                if 0 < datasets_count < 20:
                    print("获取到的数据集:")
                    for dataset in datasets_list:
                        if isinstance(dataset, dict) and 'id' in dataset:
                            dataset_name = dataset.get('name', dataset['id'])
                            print(f"  - {dataset['id']}: {dataset_name}")
                        else:
                            print(f"  - {dataset}")
                
                # 第二步：获取所有数据集的数据字段
                print("\n第二步：正在获取所有数据集的数据字段，这可能需要一些时间...")
                all_dataset_fields = self.wqb_api_client.get_all_datasets_datafields()
                
                if all_dataset_fields:
                    total_fields = sum(len(fields) for fields in all_dataset_fields.values())
                    self.logger.info(f"成功获取了 {len(all_dataset_fields)} 个数据集的共 {total_fields} 个数据字段。")
                    print(f"\n成功获取了 {len(all_dataset_fields)} 个数据集的共 {total_fields} 个数据字段。")
                    
                    # 显示每个数据集的字段数量
                    print("\n各数据集的数据字段数量:")
                    for dataset_id, fields in all_dataset_fields.items():
                        print(f"  - {dataset_id}: {len(fields)} 个字段")
                    
                    print("\n所有数据已保存到单独的文件中，避免单个文件过大。")
                else:
                    self.logger.warning("未能获取到任何数据集的数据字段。")
                    print("\n警告：未能获取到任何数据集的数据字段。")
                
            elif datasets_list is not None and len(datasets_list) == 0:
                self.logger.info("WQB平台当前未返回任何可用数据集。")
                print("\nWQB平台当前未返回任何可用数据集（或获取结果为空列表）。")
            else:
                self.logger.warning("未能获取到WQB数据集列表（API客户端返回None）。")
                print("\n未能获取到WQB数据集列表。请检查日志（可能包括API错误、网络问题或权限不足）。")

        except Exception as e:
            self.logger.error(f"更新WQB数据集和数据字段列表时发生未知错误: {e}", exc_info=True)
            print(f"\n在更新数据集和数据字段列表过程中发生错误: {e}")
            print("请检查日志获取详细信息。")

    def _view_successful_alphas(self) -> None:
        """处理用户请求查看已发现的合格Alpha。"""
        self.logger.info("用户选择查看已发现的合格Alpha。")
        print("\n--- 查看已发现的合格 Alpha ---")

        if not self.result_handler:
            self.logger.error("ResultHandler 未初始化，无法查看合格Alpha。")
            print("错误：结果处理器未准备就绪。请检查系统初始化日志。")
            return

        try:
            print("正在加载已保存的合格Alpha数据，请稍候...")
            successful_alphas_list = self.result_handler.load_successful_alphas()

            if not successful_alphas_list:
                self.logger.info("当前没有已保存的合格Alpha记录。")
                print("\n目前系统中没有已保存的合格Alpha记录。")
                return

            self.logger.info(f"成功加载了 {len(successful_alphas_list)} 条合格Alpha记录。正在格式化以便显示...")
            print(f"\n共找到 {len(successful_alphas_list)} 条合格Alpha记录。")

            alphas_df = self.result_handler.format_alphas_for_display(successful_alphas_list)

            if alphas_df.empty:
                self.logger.info("格式化后的Alpha数据显示为空。")
                print("数据加载成功，但格式化后的显示内容为空。")
            else:
                print("\n合格Alpha列表 (按计算适应度降序排列):")
                # 确保pandas的显示选项能完整展示DataFrame
                # import pandas as pd # 假设ResultHandler已导入pandas
                # pd.set_option('display.max_rows', None)
                # pd.set_option('display.max_columns', None)
                # pd.set_option('display.width', 200) # 或更大，或不设置以自适应
                print(alphas_df.to_string(index=False))

        except Exception as e:
            self.logger.error(f"查看合格Alpha时发生未知错误: {e}", exc_info=True)
            print(f"\n在查看合格Alpha过程中发生错误: {e}")
            print("请检查日志获取详细信息。")

    def _system_settings(self) -> None:
        """处理系统设置相关的用户交互和操作。"""
        self.logger.info("用户进入系统设置菜单。")

        while True:
            print("\n--- 系统设置 ---")
            print("  1. 查看当前关键配置")
            print("  2. 修改最大并发模拟数 (待实现)")
            print("  3. 修改日志级别 (待实现)")
            print("  4. 返回主菜单")
            print("-" * 20)

            choice = input("请选择设置操作 (1-4): ").strip()
            self.logger.debug(f"系统设置子菜单选择: '{choice}'")

            if choice == '1':
                self.logger.info("用户选择查看当前关键配置。")
                print("\n当前关键系统配置：")
                if self.config_manager and self.result_handler:
                    configs_to_show = {
                        "日志级别 (general.log_level)": self.config_manager.get("general.log_level", "N/A"),
                        "最大并发模拟数 (general.max_concurrent_simulations)": self.config_manager.get("general.max_concurrent_simulations", "N/A"),
                        "GA: 初始种群大小 (ga.population_size)": self.config_manager.get("ga.population_size", "N/A"),
                        "GA: 最大进化代数 (ga.max_generations)": self.config_manager.get("ga.max_generations", "N/A"),
                        "API: 基础URL (general.api_base_url)": self.config_manager.get("general.api_base_url", "N/A"),
                        "数据: 合格Alpha保存目录": self.result_handler.successful_alphas_dir,
                        "数据: GA检查点目录": self.result_handler.ga_checkpoints_dir,
                    }
                    for cfg_name, cfg_value in configs_to_show.items():
                        print(f"  - {cfg_name}: {cfg_value}")
                else:
                    print("  错误：配置管理器或结果处理器未初始化，无法完整显示配置。")
                input("\n按 Enter键 继续...")

            elif choice == '2':
                self.logger.info("用户选择修改最大并发模拟数 (功能待实现)。")
                print("\n提示：运行时修改最大并发模拟数的功能当前尚未实现。")
                print("您可以通过修改 `config/general_config.json` 文件中的 `max_concurrent_simulations` 值，")
                print("并在下次启动系统时生效。")
                input("\n按 Enter键 继续...")

            elif choice == '3':
                self.logger.info("用户选择修改日志级别 (功能待实现)。")
                print("\n提示：运行时修改日志级别的功能当前尚未实现。")
                print("您可以通过修改 `config/general_config.json` 文件中的 `log_level` 值 (例如 INFO, DEBUG, WARNING)，")
                print("并在下次启动系统时生效 (或部分立即生效，取决于日志库实现)。")
                input("\n按 Enter键 继续...")

            elif choice == '4':
                self.logger.info("用户选择从系统设置返回主菜单。")
                break
            else:
                self.logger.warning(f"系统设置中无效的选择: '{choice}'")
                print(f"无效选择 '{choice}'。请输入1到4之间的数字。")

    def _manage_blacklist(self) -> None:
        """处理API操作符/字段黑名单的管理。"""
        self.logger.info("用户进入黑名单管理菜单。")

        if not self.config_manager or not self.wqb_api_client:
            self.logger.error("ConfigManager或WQBAPIClient未初始化，无法管理黑名单。")
            print("错误：核心组件未准备就绪，无法进行黑名单管理。")
            return

        while True:
            print("\n--- API 黑名单管理 ---")
            print("  1. 查看当前黑名单")
            print("  2. 手动清除黑名单")
            print("  3. 返回主菜单")
            print("-" * 20)

            choice = input("请选择黑名单操作 (1-3): ").strip()
            self.logger.debug(f"黑名单管理子菜单选择: '{choice}'")

            if choice == '1':
                self.logger.info("用户选择查看当前黑名单。")
                print("\n当前API操作符/字段黑名单：")
                current_blacklist = self.config_manager.get("blacklist", [])
                if not current_blacklist:
                    print("  黑名单当前为空。")
                else:
                    print(f"  共找到 {len(current_blacklist)} 条黑名单记录：")
                    for idx, item in enumerate(current_blacklist):
                        name = item.get('name', 'N/A')
                        reason = item.get('reason', 'N/A')
                        timestamp = item.get('timestamp', 'N/A')
                        print(f"    {idx+1}. 名称: {name}\n       原因: {reason}\n       时间: {timestamp}")
                input("\n按 Enter键 继续...")

            elif choice == '2':
                self.logger.info("用户选择手动清除黑名单。")
                print("\n警告：清除黑名单将导致系统在后续操作中重新尝试访问之前可能已确认不可用或导致错误的操作符/字段。")
                confirm_clear = input("您确定要清除所有黑名单记录吗？(yes/no，默认为no): ").strip().lower()
                if confirm_clear == 'yes':
                    try:
                        self.wqb_api_client.clear_blacklist() # 假设此方法已在WQBAPIClient中实现
                        self.logger.info("黑名单已成功清除。ConfigManager中的记录也应被更新。")
                        print("黑名单已成功清除。")
                    except Exception as e_clear:
                        self.logger.error(f"清除黑名单时发生错误: {e_clear}", exc_info=True)
                        print(f"清除黑名单失败: {e_clear}")
                else:
                    self.logger.info("用户取消了清除黑名单的操作。")
                    print("已取消清除操作。黑名单未被修改。")
                input("\n按 Enter键 继续...")

            elif choice == '3':
                self.logger.info("用户选择从黑名单管理返回主菜单。")
                break
            else:
                self.logger.warning(f"黑名单管理中无效的选择: '{choice}'")
                print(f"无效选择 '{choice}'。请输入1到3之间的数字。")

# --- 主程序入口示例 (用于基本测试或演示) ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            stream=sys.stdout)

    orchestrator_logger = logging.getLogger(__name__ + "._main_")
    orchestrator_logger.info("开始 SystemOrchestrator 的 __main__ 测试...")

    try:
        orchestrator = SystemOrchestrator()
        orchestrator_logger.info("SystemOrchestrator 实例创建成功。")

        assert orchestrator.config_manager is not None, "ConfigManager 未初始化"
        assert orchestrator.wqb_api_client is not None, "WQB_API_Client 未初始化"
        assert orchestrator.result_handler is not None, "ResultHandler 未初始化"
        assert orchestrator.alpha_evolution_engine is not None, "AlphaEvolutionEngine 未初始化"
        assert orchestrator.result_handler.alpha_evolution_engine is orchestrator.alpha_evolution_engine, \
               "AlphaEvolutionEngine 未正确注入到 ResultHandler"
        orchestrator_logger.info("所有核心组件似乎已正确初始化并相互连接。")

        # 提示：取消以下注释以在测试初始化后直接启动主菜单循环
        # orchestrator_logger.info("准备启动主菜单循环进行手动测试...")
        # orchestrator.start()

    except RuntimeError as e:
        orchestrator_logger.critical(f"在 __main__ 测试中，SystemOrchestrator 初始化失败: {e}", exc_info=True)
    except ImportError as e_imp:
        orchestrator_logger.critical(f"在 __main__ 测试中，因导入错误导致 SystemOrchestrator 初始化失败: {e_imp}", exc_info=True)
        orchestrator_logger.critical("请确保所有依赖模块...都存在于正确的路径下...")
    except Exception as e_other:
        orchestrator_logger.critical(f"在 __main__ 测试中发生未知错误: {e_other}", exc_info=True)

    orchestrator_logger.info("SystemOrchestrator 的 __main__ 测试结束。")

