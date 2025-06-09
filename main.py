# main.py
# 系统主入口文件

import logging # 用于在Orchestrator实例化前后记录顶层信息
import sys # 用于在导入失败时安全退出
import traceback # 后续用于打印完整异常堆栈

# 尝试导入核心编排器和日志设置工具
# 这里的 try-except 更侧重于指示用户环境配置问题
try:
    from src.system_orchestrator import SystemOrchestrator
    # setup_logging 会在 SystemOrchestrator 内部被调用，这里不需要直接导入
    # from src.utils import setup_logging
except ImportError as e:
    # 在非常早期的导入失败，可能意味着项目结构问题或PYTHONPATH未正确设置
    # 此时日志系统可能还未初始化，直接打印到stderr
    print(f"错误：无法导入核心系统模块 (SystemOrchestrator): {e}", file=sys.stderr)
    print("请确保项目结构正确，并且src目录在PYTHONPATH中（如果需要），或者您正在从项目根目录运行此脚本。", file=sys.stderr)
    print("例如，使用 'python main.py' 或 'python -m main' (如果main是包的一部分)。", file=sys.stderr)
    sys.exit(1) # 关键模块无法导入，系统无法启动
except Exception as e_other_import: #捕获其他可能的导入时异常
    print(f"错误：导入模块时发生非预期的错误: {e_other_import}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    # 此时，utils.setup_logging() 还未被 SystemOrchestrator._initialize_components() 调用。
    # 如果希望在 SystemOrchestrator.__init__ 之前就有格式化的日志，
    # 可以在这里进行一次临时的、基础的日志配置，但这通常不是推荐做法，
    # 因为配置可能与 setup_logging 中的最终配置冲突。
    # 我们将依赖 SystemOrchestrator 内部调用 setup_logging() 来完成标准日志配置。

    # 获取一个临时的顶层logger，主要用于记录main.py本身的启动和退出事件。
    # 其格式化和级别将依赖于 setup_logging 何时被调用并配置根logger。
    # 在 setup_logging 调用前，它可能使用Python logging的默认配置。
    main_logger = logging.getLogger("main_entry_point")
    # 为了确保至少能看到 main_logger 的信息，可以设置一个基本级别，
    # 但这会被 setup_logging 覆盖。
    # main_logger.setLevel(logging.INFO)
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # main_logger.addHandler(handler)
    # main_logger.propagate = False # 如果根logger已有其他配置，避免重复输出

    main_logger.info("系统主入口 (main.py) 开始执行...")
    main_logger.info("准备实例化 SystemOrchestrator 并启动系统...")

    orchestrator: SystemOrchestrator # 类型提示
    try:
        orchestrator = SystemOrchestrator()
        # SystemOrchestrator 的 __init__ -> _initialize_components 会调用 setup_logging()
        # 所以从这里开始，日志应该会按照配置文件中的设置进行格式化和输出。
        # （除非 setup_logging 失败，那种情况下 _initialize_components 会有相应处理）

        main_logger.info("SystemOrchestrator 实例化成功，准备调用 start()...")
        orchestrator.start() # 启动系统的主控制流程

    except RuntimeError as e_runtime:
        # 此异常通常由 SystemOrchestrator.__init__ 在核心组件初始化失败时抛出
        main_logger.critical(f"系统启动失败，发生运行时错误: {e_runtime}", exc_info=True)
        # 错误消息通常已在 SystemOrchestrator 的 __init__ 或其调用的方法中打印
        # 如果 SystemOrchestrator 的 __init__ 重新抛出了原始异常，
        # 这里的 exc_info=True 会记录那个原始异常的堆栈。
        # 如果 SystemOrchestrator 的 start() 方法中的 sys.exit(1) 被执行，
        # 这里的 sys.exit 可能不会被达到。
        sys.exit(1)
    except ImportError as e_import_runtime:
        # 这个 except 块是为了捕获在 SystemOrchestrator 实例化过程中，
        # 如果其内部的 try-except ImportError 逻辑由于某种原因再次失败
        # (例如，PYTHONPATH在运行时被修改，或者某些动态导入失败)
        main_logger.critical(f"系统启动因运行时导入错误而失败: {e_import_runtime}", exc_info=True)
        print(f"错误：运行时无法导入必要的模块: {e_import_runtime}", file=sys.stderr)
        print("请检查项目依赖和PYTHONPATH配置。", file=sys.stderr)
        sys.exit(1)
    except Exception as e_unexpected:
        # 捕获其他在Orchestrator实例化或start()调用前/过程中发生的意外错误
        main_logger.critical(f"在系统启动过程中发生未预期的严重错误: {e_unexpected}", exc_info=True)
        # 使用 traceback.print_exc() 确保即使日志系统配置不完整，也能打印堆栈
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        main_logger.info("系统主入口 (main.py) 执行完毕。系统已退出或尝试退出。")

