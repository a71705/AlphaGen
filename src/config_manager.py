# -*- coding: utf-8 -*-
import json
import os
import logging

class ConfigManager:
    """
    配置管理器类，用于加载和管理应用程序的配置。
    这是一个单例类，确保在整个应用程序中只有一个配置管理器实例。
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式。如果实例不存在，则创建一个新实例；否则，返回现有实例。
        """
        if cls._instance is None:
            # 使用 Python 3 的 super() 调用方式
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_base_path="config/", data_base_path="data/"):
        """
        初始化配置管理器。

        Args:
            config_base_path (str): 配置文件存放的基础路径。
            data_base_path (str): 数据文件存放的基础路径。
        """
        if self._initialized:
            return

        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化 ConfigManager...")

        self.config_base_path = config_base_path
        self.data_base_path = data_base_path
        self.config = {}

        # 确保基础路径存在
        if not os.path.exists(self.config_base_path):
            os.makedirs(self.config_base_path)
            self.logger.info(f"配置路径 {self.config_base_path} 不存在，已创建。")
        if not os.path.exists(self.data_base_path):
            os.makedirs(self.data_base_path)
            self.logger.info(f"数据路径 {self.data_base_path} 不存在，已创建。")

        self._load_all_configs() # 加载所有配置
        self._initialized = True
        self.logger.info("ConfigManager 初始化完成。")

    def _load_json_file(self, file_path: str) -> dict:
        """
        加载并解析一个JSON文件。

        Args:
            file_path (str): 要加载的JSON文件的完整路径。

        Returns:
            dict: 解析后的JSON数据。如果文件未找到，则返回空字典。

        Raises:
            json.JSONDecodeError: 如果JSON文件格式无效。
            Exception: 如果发生其他未预期的错误。
        """
        try:
            # 尝试以只读模式和UTF-8编码打开并读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"成功加载并解析JSON文件: {file_path}")
            return data
        except FileNotFoundError:
            # 文件未找到，记录警告并返回空字典
            self.logger.warning(f"JSON文件未找到: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            # JSON解析失败，记录错误并重新抛出异常
            self.logger.error(f"解析JSON文件失败: {file_path} - 错误: {e}")
            raise  # 重新抛出原始异常以保留堆栈跟踪
        except Exception as e:
            # 捕获其他潜在的异常，记录错误并重新抛出
            self.logger.error(f"加载JSON文件 {file_path} 时发生未知错误: {e}")
            raise # 重新抛出原始异常

    def _load_all_configs(self):
        """
        加载所有指定的JSON配置文件到 self.config 字典中。

        该方法会依次尝试加载以下文件:
        - 来自 config_base_path:
            - general_config.json (键: "general")
            - alpha_templates.json (键: "alpha_templates")
            - operator_sets.json (键: "operator_sets")
            - ga_config.json (键: "ga")
        - 来自 data_base_path:
            - available_datafields.json (键: "datafields")
            - inaccessible_ops_blacklist.json (键: "blacklist")

        如果某个文件加载失败或未找到，`_load_json_file` 会记录相应的日志，
        并在 self.config 中为对应的键设置一个空字典。
        """
        self.logger.info("开始加载所有配置文件...")

        # 定义配置文件及其对应的键和基础路径
        config_files_to_load = [
            # (文件名, 配置键, 基础路径属性名)
            ("general_config.json", "general", self.config_base_path),
            ("alpha_templates.json", "alpha_templates", self.config_base_path),
            ("operator_sets.json", "operator_sets", self.config_base_path),
            ("ga_config.json", "ga", self.config_base_path),
            ("available_datafields.json", "datafields", self.data_base_path),
            ("inaccessible_ops_blacklist.json", "blacklist", self.data_base_path),
        ]

        for filename, config_key, base_path in config_files_to_load:
            file_path = os.path.join(base_path, filename)
            self.config[config_key] = self._load_json_file(file_path)
            # _load_json_file 内部会记录加载成功/失败的日志
            # 如果文件未找到，会返回 {} 并记录警告

        self.logger.info("所有配置文件加载尝试完毕。")

    def _save_config_file(self, file_path: str, data) -> bool:
        """
        将数据（字典或列表）以JSON格式保存到指定文件路径。

        Args:
            file_path (str): 要保存的文件的完整路径。
            data (Union[dict, list]): 要序列化为JSON并保存的Python字典或列表。

        Returns:
            bool: 如果保存成功则返回 True，否则返回 False。
        """
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 以写入模式和UTF-8编码打开文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            self.logger.debug(f"配置文件已成功保存到: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存配置文件到 {file_path} 失败: {e}")
            return False

    def get(self, key_path: str, default=None):
        """
        从加载的配置中获取指定路径的值。

        Args:
            key_path (str): 配置项的路径，使用点（.）分隔。
                            例如："general.api_url" 或 "ga.settings.population_size"。
            default: 如果找不到指定的配置项，返回的默认值。默认为 None。

        Returns:
            any: 找到的配置值。如果路径无效或键不存在，则返回 `default` 参数指定的值。
        """
        if not isinstance(self.config, dict) or not self.config:
            self.logger.warning(f"配置字典为空或未正确初始化，无法获取路径 '{key_path}'。")
            return default

        keys = key_path.split('.')
        current_level_data = self.config

        for i, key in enumerate(keys):
            try:
                # 检查当前层级是否为字典类型，并且键是否存在
                if isinstance(current_level_data, dict) and key in current_level_data:
                    current_level_data = current_level_data[key]
                else:
                    # 如果中间路径的某个键不存在，或者current_level_data不是字典了
                    partial_path = '.'.join(keys[:i+1])
                    self.logger.warning(f"在配置路径 '{key_path}' 中未找到键 '{key}' (在子路径 '{partial_path}' 处)。返回默认值。")
                    return default
            except TypeError:
                # 如果current_level_data不是可索引的 (例如，它是一个列表或简单类型，但我们期望一个字典)
                partial_path = '.'.join(keys[:i]) # 路径直到前一个键
                self.logger.warning(f"在配置路径 '{key_path}' 查找时，路径 '{partial_path}' 的值不是预期的字典类型。返回默认值。")
                return default
            except Exception as e:
                # 捕获其他潜在错误
                self.logger.error(f"获取配置路径 '{key_path}' 时发生未知错误: {e}。返回默认值。")
                return default

        return current_level_data

    def set_datafields(self, datafields: list):
        """
        更新内存中的可用数据字段列表，并将其持久化保存到JSON文件。

        Args:
            datafields (list): 新的可用数据字段列表。

        Returns:
            None
        """
        if not isinstance(datafields, list):
            self.logger.error("set_datafields 接收到的参数不是列表类型。")
            return

        self.config['datafields'] = datafields  # 更新内存中的配置

        file_path = os.path.join(self.data_base_path, "available_datafields.json")

        if self._save_config_file(file_path, datafields):
            self.logger.info(f"可用数据字段列表已更新并保存，共 {len(datafields)} 个条目。")
        else:
            self.logger.error("可用数据字段列表更新后，保存到文件失败。")

    def set_blacklist(self, blacklist: list):
        """
        更新内存中的不可访问操作符黑名单，并将其持久化保存到JSON文件。

        Args:
            blacklist (list): 新的不可访问操作符黑名单列表。

        Returns:
            None
        """
        if not isinstance(blacklist, list):
            self.logger.error("set_blacklist 接收到的参数不是列表类型。")
            return

        self.config['blacklist'] = blacklist  # 更新内存中的配置

        file_path = os.path.join(self.data_base_path, "inaccessible_ops_blacklist.json")

        if self._save_config_file(file_path, blacklist):
            self.logger.info(f"不可访问操作符黑名单已更新并保存，共 {len(blacklist)} 个条目。")
        else:
            self.logger.error("不可访问操作符黑名单更新后，保存到文件失败。")

if __name__ == '__main__':
    # 此部分用于基本测试，后续可以移除或移至测试文件
    logging.basicConfig(level=logging.INFO)

    # 获取 ConfigManager 实例
    cm1 = ConfigManager()
    cm1.logger.info(f"CM1 Config Base Path: {cm1.config_base_path}")
    cm1.logger.info(f"CM1 Data Base Path: {cm1.data_base_path}")
    cm1.logger.info(f"CM1 Configs: {cm1.config}")

    # 再次获取实例，验证是否为单例
    cm2 = ConfigManager(config_base_path="new_config/", data_base_path="new_data/")
    cm2.logger.info(f"CM2 Config Base Path: {cm2.config_base_path}") # 应仍为 "config/"
    cm2.logger.info(f"CM2 Data Base Path: {cm2.data_base_path}")   # 应仍为 "data/"

    if id(cm1) == id(cm2):
        cm1.logger.info("ConfigManager 是单例。")
    else:
        cm1.logger.error("ConfigManager 不是单例。")

    # 尝试加载一个不存在的配置（仅为演示 _load_all_configs 的预期行为）
    # cm1._load_all_configs() # 假设它尝试加载但文件不存在
    cm1.logger.info(f"CM1 Configs after attempting load: {cm1.config}")
