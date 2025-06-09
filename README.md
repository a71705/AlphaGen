# WorldQuant Alpha 因子自动化挖掘系统

**版本:** 1.0 (此版本基于截至Task T7.0的开发状态)
**最后更新:** 2024年06月16日

## 1. 项目概览

本项目旨在构建一个高度自动化、智能、灵活、鲁棒且内存高效的 WorldQuant Brain Alpha 因子（投资策略）发现平台。系统利用遗传编程 (Genetic Programming, GP) 原理，与 WorldQuant Brain (WQB) API 集成，用于全自动地生成金融Alpha表达式、提交模拟、评估性能、并筛选出符合预设标准的优质Alpha因子。

核心功能包括：
- 基于JSON配置文件的全系统参数化管理。
- 与WQB API的健壮交互（认证、会话管理、带重试的API调用）。
- 基于模板和初步结构化方法的Alpha表达式（AST形式）生成。
- 可配置的Alpha性能适应度计算。
- 通过遗传算法（选择、交叉占位符、变异占位符）进行Alpha种群的进化。
- 并发执行Alpha模拟以提高效率。
- 详细的模拟结果处理、数据提取与持久化（合格Alpha详情、PnL数据、年度统计）。
- GA状态的保存与加载，支持断点续传。
- 通过命令行界面进行系统操作和监控。

## 2. 技术栈

- **Python 3.x** (建议 3.9 或更高版本，以支持现代类型提示等特性)
- **requests**: 用于与 WorldQuant Brain HTTP API 进行交互。
- **pandas**: 用于处理和分析表格化数据，如Alpha的PnL序列和年度统计数据。
- **pyparsing**: 用于Alpha表达式（基于WQB的FASTEXPR语法）的解析，将其从字符串形式转换为抽象语法树 (AST)，并支持从AST生成表达式字符串。

## 3. 项目结构

项目的目录和文件结构如下：

```
/project_root/
├── config/                     # 存放所有JSON格式的系统配置文件
│   ├── general_config.json     # 通用配置 (API URL, 并发数, 日志级别, WQB凭证文件路径等)
│   ├── alpha_templates.json    # Alpha表达式模板定义 (用于基于模板的Alpha生成)
│   ├── operator_sets.json      # 可用的操作符、数据字段常量、参数取值范围等
│   └── ga_config.json          # 遗传算法参数 (种群大小, 适应度函数权重, 进化代数等)
├── data/                       # 存放运行时生成的数据、GA检查点和Alpha结果
│   ├── ga_checkpoints/         # GA状态检查点 (JSON格式, 用于断点续传)
│   ├── successful_alphas/      # 存储被判定为合格的Alpha的详细信息 (JSON格式)
│   ├── alpha_pnl/              # (若配置获取) 存储合格Alpha的PnL数据 (CSV格式)
│   ├── alpha_yearly_stats/     # (若配置获取) 存储合格Alpha的年度统计数据 (CSV格式)
│   ├── available_datafields.json # WQB平台可用的数据字段列表 (系统可自动更新)
│   └── inaccessible_ops_blacklist.json # API操作中遇到的不可访问操作符/字段黑名单 (系统动态更新)
├── logs/                       # 存放运行时日志文件
│   └── YYYY-MM-DD/             # 日志按日期分子目录组织
│       ├── system.log          # 通用系统日志 (记录INFO及以上级别)
│       └── errors.log          # 错误日志 (专门记录ERROR及以上级别)
├── src/                        # 项目的核心源代码模块
│   ├── config_manager.py       # 配置管理器模块 (ConfigManager类)
│   ├── wqb_api_client.py       # WQB API客户端模块 (WQB_API_Client类)
│   ├── alpha_evolution_engine.py # Alpha进化引擎模块 (AlphaEvolutionEngine类, Individual类)
│   ├── result_handler.py       # 模拟结果处理与数据持久化模块 (ResultHandler类)
│   ├── system_orchestrator.py  # 系统总控制器与用户交互模块 (SystemOrchestrator类)
│   └── utils.py                # 通用工具函数 (日志设置, AST转换工具等)
├── main.py                     # 系统的主启动入口脚本
├── requirements.txt            # Python第三方依赖库列表
├── 开发文档.md                 # 详细的项目开发与设计文档 (简体中文)
└── README.md                   # 本文件，提供项目概览和使用指南
```

## 4. 安装与配置

### 4.1. 环境准备
- 确保已安装 Python 3.x (建议 3.9+)。
- (推荐) 创建并激活一个Python虚拟环境：
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  # venv\Scriptsctivate   # Windows
  ```

### 4.2. 克隆仓库
```bash
git clone <repository_url>  # 请替换为实际的仓库URL
cd <project_root_directory_name> # 请替换为实际的项目根目录名
```

### 4.3. 安装依赖
```bash
pip install -r requirements.txt
```
请确保 `requirements.txt` 文件包含以下核心依赖（及它们各自的依赖）：
```
requests
pandas
pyparsing
```

### 4.4. 系统配置
所有系统配置均通过 `config/` 目录下的JSON文件进行管理。

1.  **WQB API凭证配置 (必需)**:
    *   在 `config/` 目录下，创建名为 `credentials.json` 的文件。
    *   文件内容应为JSON格式，包含您的 WorldQuant Brain 邮箱和密码：
        ```json
        {
            "email": "YOUR_WQB_EMAIL@example.com",
            "password": "YOUR_WQB_PASSWORD"
        }
        ```
    *   **重要**: `config/general_config.json` 文件中的 `credentials_file` 字段定义了此凭证文件的路径 (默认为 `"config/credentials.json"`)。请务必保护好此凭证文件。

2.  **通用配置 (`config/general_config.json`)**:
    *   包含API基础URL、会话文件路径、日志级别、最大并发模拟数、API请求重试参数、可选数据获取开关（如PnL、年度统计）以及各类数据存储子目录的名称等。

3.  **Alpha模板配置 (`config/alpha_templates.json`)**:
    *   定义了用于基于模板生成Alpha的表达式模板及其占位符规则。

4.  **操作符集配置 (`config/operator_sets.json`)**:
    *   定义了可用于Alpha生成的各类操作符、数据字段常量以及操作符参数的可选值范围。

5.  **遗传算法配置 (`config/ga_config.json`)**:
    *   定义了遗传算法的核心参数，如种群大小、最大进化代数、精英选择数量、交叉与变异率、适应度函数计算方式（指标权重、惩罚因子等）以及哪些模拟参数参与遗传进化。

请根据您的具体需求和WQB平台特性，仔细检查并调整这些配置文件。通常，**除了 `credentials.json` 必须由用户创建和填写外**，其他配置文件可以使用项目提供的默认值开始运行。

## 5. 运行系统

在完成安装和配置后，从项目的根目录运行 `main.py` 脚本来启动系统：

```bash
python main.py
```

系统启动后，将显示一个基于文本的主菜单。根据菜单提示输入相应的数字选项来执行操作：

-   **选项 1: 启动 Alpha 挖掘 (遗传优化)**: 开始新的Alpha发现任务或从之前的检查点恢复。
-   **选项 2: 爬取/更新 WQB 数据字段列表**: 从WQB平台同步最新的可用数据字段。
-   **选项 3: 查看已发现的合格 Alpha**: 展示已保存的、符合标准的Alpha策略摘要。
-   **选项 4: 系统设置**: 查看部分当前系统配置（运行时修改功能多为占位符，提示通过修改JSON文件实现）。
-   **选项 5: 黑名单管理**: 查看或清除在API交互中动态生成的不可访问资源黑名单。
-   **选项 6: 退出系统**: 安全关闭应用程序。

所有操作过程和重要事件都会被记录到 `logs/` 目录下的对应日期的日志文件中。

## 6. 模块说明

-   **`src/utils.py`**: 提供通用工具，包括日志系统初始化 (`setup_logging`) 和Alpha表达式的AST解析与生成工具 (`Node` 类, `parse_expression_to_ast`, `generate_expression_from_ast`)。
-   **`src/config_manager.py`**: (`ConfigManager` 类) 单例模式，负责加载、管理所有JSON配置文件，并支持部分配置的动态更新与持久化。
-   **`src/wqb_api_client.py`**: (`WQB_API_Client` 类) 封装与WQB API的全部交互，包括认证（含Persona流程）、会话管理、带重试的核心请求逻辑、具体API功能调用及黑名单管理。支持异步模拟提交。
-   **`src/alpha_evolution_engine.py`**: (`AlphaEvolutionEngine` 类, `Individual` 类) GA核心引擎，负责Alpha个体的表示、种群管理、Alpha表达式生成（模板和初步结构化）、适应度计算框架、遗传操作初步框架（AST操作为占位符）及进化流程控制。
-   **`src/result_handler.py`**: (`ResultHandler` 类) 处理Alpha模拟结果，提取指标，协调适应度计算，判断合格性，持久化合格Alpha数据（JSON详情、CSV的PnL和年度统计），并管理GA状态的保存与加载。
-   **`src/system_orchestrator.py`**: (`SystemOrchestrator` 类) 系统总控制器，初始化各模块，处理依赖（含循环依赖），并提供用户交互的主菜单界面。
-   **`main.py`**: 系统的主启动入口脚本。

## 7. 开发文档参考

更详细的系统设计、模块职责、接口定义和各开发任务（Task Cards）的详细说明，请参阅项目根目录下的 `开发文档.md` 文件。该文档是理解系统内部工作原理和进行后续开发的重要参考。

## 8. 注意事项与未来工作

-   **AST的交叉与变异**: `AlphaEvolutionEngine` 中对Alpha表达式AST的交叉和变异操作当前仅为初步框架/占位符。后续需要实现更成熟的遗传编程操作。
-   **配置项的运行时修改**: `SystemOrchestrator` 菜单中修改配置的选项多为占位符。未来可增强 `ConfigManager` 以支持更多配置的运行时修改和持久化。
-   **测试覆盖**: 当前各模块的 `if __name__ == '__main__':` 块提供了基本测试。为确保系统质量，未来应引入如 `pytest` 等专用测试框架，编写更全面的单元测试和集成测试。
-   **错误处理与鲁棒性**: 尽管已努力加入错误处理，但实际运行中可能遇到更多API响应或边界情况，需持续完善。
-   **性能优化**: 对于大规模长时间运行的GA，可能需进一步分析和优化性能瓶颈。

---
## 许可证

(请在此处根据项目的实际情况填写许可证信息，例如：本项目根据 MIT 许可证授权。)
```
