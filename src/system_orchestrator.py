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
    # Fallback for scenarios where relative imports might fail
    # (e.g., direct script execution without proper package setup, or different project structure)
    # This assumes these modules are in the PYTHONPATH or current working directory.
    # Log the fact that we're using fallback imports, this can be helpful for debugging import issues.
    # Note: logging might not be configured yet if setup_logging itself fails to import this way.
    print(f"SystemOrchestrator: Relative import failed ('{e}'), attempting absolute import fallback.", file=sys.stderr)
    try:
        from config_manager import ConfigManager
        from wqb_api_client import WQB_API_Client
        from alpha_evolution_engine import AlphaEvolutionEngine
        from result_handler import ResultHandler
        from utils import setup_logging
    except ImportError as e_abs:
        # If absolute imports also fail, this is a critical issue.
        print(f"CRITICAL: SystemOrchestrator - Absolute import failed for core modules: {e_abs}. System cannot start.", file=sys.stderr)
        # Re-raise the last error or a custom one to halt execution if this is a library
        # For a main script, sys.exit might be appropriate here.
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
        # 获取一个logger实例。此时全局的setup_logging可能还未被调用，
        # 所以这里的日志可能使用logging的默认基本配置。
        # setup_logging的调用被放在_initialize_components的第一步。
        self.logger = logging.getLogger(__name__)
        self.logger.info("SystemOrchestrator 实例化开始...")

        # 定义核心组件的实例变量，并进行类型提示 (初始为None)
        self.config_manager: ConfigManager = None # type: ignore
        self.wqb_api_client: WQB_API_Client = None # type: ignore
        self.alpha_evolution_engine: AlphaEvolutionEngine = None # type: ignore
        self.result_handler: ResultHandler = None # type: ignore

        try:
            self._initialize_components()
            self.logger.info("SystemOrchestrator 核心组件初始化成功。")
        except Exception as e:
            # 使用 self.logger 如果它已经被 _initialize_components 中的 setup_logging 配置过
            # 否则，它可能还是原始的 logger。也可以直接用 print 和 traceback。
            critical_msg = f"SystemOrchestrator 初始化核心组件时发生致命错误: {e}"
            if hasattr(self, 'logger') and self.logger.handlers: # Check if logger has handlers (configured)
                self.logger.critical(critical_msg, exc_info=True)
            else: # Fallback to print if logger is not yet configured
                print(critical_msg, file=sys.stderr)
                traceback.print_exc()

            # 考虑到这是构造函数，如果初始化失败，系统无法继续。
            # 重新抛出异常，让调用者知道初始化失败。
            raise RuntimeError(f"系统核心组件初始化失败: {e}") from e
            # 或者，如果这是一个顶层应用，可以考虑 sys.exit(1)
            # print(f"错误：系统核心组件初始化失败 - {e}")
            # print("请检查日志获取详细信息。系统可能无法正常运行。")


    def _initialize_components(self) -> None:
        """
        私有方法：初始化所有系统核心组件并处理它们之间的依赖关系。
        """
        # 1. 初始化日志系统 (这是第一步，确保后续所有日志都已配置)
        try:
            setup_logging() # 调用utils中的日志设置函数
            # 更新 self.logger 以使用配置好的logger
            # (getLogger(__name__)会获取到基于新配置的logger)
            self.logger = logging.getLogger(__name__)
            self.logger.info("全局日志系统已初始化。")
        except Exception as log_e:
            # 如果日志系统初始化失败，这是个严重问题。
            print(f"严重错误：日志系统初始化失败！{log_e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # 后续的日志可能不会按预期工作，但程序仍会尝试继续其他组件的初始化。
            # 或者，可以决定在这里抛出异常终止程序。
            # raise RuntimeError(f"日志系统初始化失败: {log_e}") from log_e

        # 2. 实例化ConfigManager
        self.logger.debug("正在实例化 ConfigManager...")
        try:
            self.config_manager = ConfigManager()
            self.logger.info("ConfigManager 已实例化。")
        except Exception as e:
            self.logger.critical(f"ConfigManager 实例化失败: {e}", exc_info=True)
            raise # ConfigManager是基础，失败则无法继续

        # 3. 根据ConfigManager的配置更新日志级别
        # (假设 setup_logging() 可能设置了默认级别，这里根据配置再调整一次)
        try:
            log_level_str = self.config_manager.get("general.log_level", "INFO").upper()
            numeric_log_level = getattr(logging, log_level_str, logging.INFO)
            # 获取根logger并设置级别，这将影响所有子logger除非它们有自己的特定级别设置
            logging.getLogger().setLevel(numeric_log_level)
            self.logger.info(f"全局日志级别已根据配置更新为: {log_level_str} ({numeric_log_level})")
        except Exception as e:
            self.logger.warning(f"根据配置更新日志级别失败: {e}. 将使用 setup_logging 设定的级别或默认级别。", exc_info=True)


        # 4. 实例化WQB_API_Client
        self.logger.debug("正在实例化 WQB_API_Client...")
        try:
            self.wqb_api_client = WQB_API_Client(config_manager=self.config_manager)
            self.logger.info("WQB_API_Client 已实例化。")
        except Exception as e:
            self.logger.critical(f"WQB_API_Client 实例化失败: {e}", exc_info=True)
            raise

        # 5. 实例化ResultHandler (此时alpha_evolution_engine为None)
        self.logger.debug("正在实例化 ResultHandler (alpha_evolution_engine 暂为 None)...")
        try:
            self.result_handler = ResultHandler(
                config_manager=self.config_manager,
                wqb_api_client=self.wqb_api_client,
                alpha_evolution_engine_ref=None # 明确传递None
            )
            self.logger.info("ResultHandler 已实例化。")
        except Exception as e:
            self.logger.critical(f"ResultHandler 实例化失败: {e}", exc_info=True)
            raise

        # 6. 实例化AlphaEvolutionEngine
        self.logger.debug("正在实例化 AlphaEvolutionEngine...")
        try:
            self.alpha_evolution_engine = AlphaEvolutionEngine(
                config_manager=self.config_manager,
                wqb_api_client=self.wqb_api_client,
                result_handler=self.result_handler # ResultHandler实例已创建
            )
            self.logger.info("AlphaEvolutionEngine 已实例化。")
        except Exception as e:
            self.logger.critical(f"AlphaEvolutionEngine 实例化失败: {e}", exc_info=True)
            raise

        # 7. 解决循环依赖：将AlphaEvolutionEngine实例注入到ResultHandler中
        self.logger.debug("正在将 AlphaEvolutionEngine 实例注入到 ResultHandler...")
        try:
            self.result_handler.set_alpha_evolution_engine(self.alpha_evolution_engine)
            self.logger.info("AlphaEvolutionEngine 实例已成功注入 ResultHandler。")
        except Exception as e:
            self.logger.error(f"将 AlphaEvolutionEngine 注入 ResultHandler 时失败: {e}", exc_info=True)
            # 这是一个潜在的问题，但不一定是致命到需要立即停止整个系统，取决于后续操作
            # 但如果 ResultHandler 强依赖此引用，后续会出错。

        # (可选) 如果AlphaEvolutionEngine也需要ResultHandler的后期注入，则在这里调用
        # if hasattr(self.alpha_evolution_engine, 'set_result_handler'):
        #     self.logger.debug("正在将 ResultHandler 实例注入到 AlphaEvolutionEngine...")
        #     self.alpha_evolution_engine.set_result_handler(self.result_handler) # type: ignore
        #     self.logger.info("ResultHandler 实例已成功注入 AlphaEvolutionEngine。")

        self.logger.info("所有核心系统组件均已成功初始化和配置。")

# --- 主程序入口示例 (用于基本测试或演示) ---
if __name__ == '__main__':
    # 注意：直接运行此文件可能因相对导入问题而失败，除非项目结构和PYTHONPATH配置正确
    # 或者在顶层 main.py 中实例化 SystemOrchestrator

    # 临时的基本日志配置，以便在直接运行此文件进行测试时能看到一些输出
    # 真正的日志配置应该由 utils.setup_logging() 完成
    if not logging.getLogger().hasHandlers(): # 避免重复添加处理器
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            stream=sys.stdout) # 确保输出到stdout

    orchestrator_logger = logging.getLogger(__name__ + "._main_")
    orchestrator_logger.info("开始 SystemOrchestrator 的 __main__ 测试...")

    # 为了能独立运行和测试 SystemOrchestrator 的初始化，
    # 我们需要确保依赖的模块（ConfigManager等）能够被找到。
    # 如果是在一个完整的项目结构中，并且PYTHONPATH设置正确，或者通过 `python -m src.system_orchestrator` 运行，
    # 那么顶部的 try-except 导入应该能处理。
    # 如果直接 `python src/system_orchestrator.py`，则需要确保 utils.py, config_manager.py 等在同一目录或PYTHONPATH中。

    # 假设依赖的模块和 utils.setup_logging() 能够正常工作：
    try:
        orchestrator = SystemOrchestrator()
        orchestrator_logger.info("SystemOrchestrator 实例创建成功。")

        # 进行一些基本断言来验证组件是否被初始化
        assert orchestrator.config_manager is not None, "ConfigManager 未初始化"
        orchestrator_logger.debug(f"ConfigManager instance: {type(orchestrator.config_manager)}")

        assert orchestrator.wqb_api_client is not None, "WQB_API_Client 未初始化"
        orchestrator_logger.debug(f"WQBAPIClient instance: {type(orchestrator.wqb_api_client)}")

        assert orchestrator.result_handler is not None, "ResultHandler 未初始化"
        orchestrator_logger.debug(f"ResultHandler instance: {type(orchestrator.result_handler)}")

        assert orchestrator.alpha_evolution_engine is not None, "AlphaEvolutionEngine 未初始化"
        orchestrator_logger.debug(f"AlphaEvolutionEngine instance: {type(orchestrator.alpha_evolution_engine)}")

        assert orchestrator.result_handler.alpha_evolution_engine is orchestrator.alpha_evolution_engine, \
               "AlphaEvolutionEngine 未正确注入到 ResultHandler"
        orchestrator_logger.info("所有核心组件似乎已正确初始化并相互连接。")

    except RuntimeError as e:
        orchestrator_logger.critical(f"在 __main__ 测试中，SystemOrchestrator 初始化失败: {e}", exc_info=True)
    except ImportError as e_imp:
        orchestrator_logger.critical(f"在 __main__ 测试中，因导入错误导致 SystemOrchestrator 初始化失败: {e_imp}", exc_info=True)
        orchestrator_logger.critical("请确保所有依赖模块 (ConfigManager, WQB_API_Client, AlphaEvolutionEngine, ResultHandler, utils.setup_logging) "
                                     "都存在于正确的路径下，并且项目结构配置正确以支持模块导入。")
    except Exception as e_other:
        orchestrator_logger.critical(f"在 __main__ 测试中发生未知错误: {e_other}", exc_info=True)

    orchestrator_logger.info("SystemOrchestrator 的 __main__ 测试结束。")

```
