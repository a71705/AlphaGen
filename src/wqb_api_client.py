# -*- coding: utf-8 -*-
import requests
import os
import json # 需要用于凭证加载
import pickle
import time # 需要用于重试逻辑中的 sleep
import logging
import re # 需要用于从错误消息中提取信息
from urllib.parse import urljoin
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

# 假设 ConfigManager 类存在于 src.config_manager
# from src.config_manager import ConfigManager # 实际项目中需要正确的导入路径

class WQB_API_Client:
    """
    WQB (WorldQuant BRAIN) API 客户端类。
    负责与WQB API进行交互，包括认证、数据获取、Alpha提交与模拟等。
    """

    def __init__(self, config_manager):
        """
        初始化 WQB_API_Client。

        Args:
            config_manager: ConfigManager类的实例，用于获取配置信息。
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化 WQB_API_Client...")

        self.config_manager = config_manager

        # 1. 初始化会话 (存根调用)
        self.session: requests.Session = self._initialize_session()
        self.logger.info("Session 对象已初始化 (通过存根)。")

        # 2. 从 ConfigManager 获取配置
        self.base_url = self.config_manager.get("general.api_base_url", "https://api.worldquantbrain.com/") # 提供默认值
        self.credentials_file = self.config_manager.get("general.credentials_file", "config/credentials.json")
        self.session_pkl_file = self.config_manager.get("general.session_pkl_file", "data/session.pkl")

        # 用于批处理监控执行器的最大工作线程数
        max_monitor_workers = self.config_manager.get("general.max_concurrent_simulations", 5) # 默认5个

        # 获取黑名单，如果配置中不存在或为空，则默认为空列表
        self._inaccessible_ops_blacklist = self.config_manager.get("blacklist", [])
        if not isinstance(self._inaccessible_ops_blacklist, list):
            self.logger.warning(f"配置中的 'blacklist' 不是列表类型，已重置为空列表。收到的类型: {type(self._inaccessible_ops_blacklist)}")
            self._inaccessible_ops_blacklist = []

        self.logger.info(f"API Base URL: {self.base_url}") # 记录API基础URL
        self.logger.info(f"Credentials File: {self.credentials_file}") # 记录凭证文件路径
        self.logger.info(f"Session Pickle File: {self.session_pkl_file}") # 记录会话pickle文件路径
        self.logger.info(f"Max Concurrent Simulations for Monitor: {max_monitor_workers}") # 记录最大并发监控模拟任务数
        self.logger.info(f"Inaccessible Ops Blacklist条目数: {len(self._inaccessible_ops_blacklist)}") # 记录不可访问操作黑名单的条目数

        # 3. 构建API端点URL
        self.login_endpoint = urljoin(self.base_url, "authentication") # 登录认证API端点
        self.alpha_endpoint = urljoin(self.base_url, "alphas") # Alpha相关API端点
        self.simulation_endpoint = urljoin(self.base_url, "simulations") # 模拟相关API端点
        self.data_fields_endpoint = urljoin(self.base_url, "data-fields") # 数据字段API端点
        self.data_sets_endpoint = urljoin(self.base_url, "data-sets") # 数据集API端点
        self.logger.info("API 端点已构建。") # 记录API端点构建完成

        # 4. 初始化批处理相关属性
        self._batch_queue = []
        self._batch_queue_lock = threading.Lock()
        # 批处理提交执行器，通常只需要一个线程来序列化提交到队列的操作
        self._batch_submit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='BatchSubmitThread')
        # 批处理监控执行器，使用从配置中获取的并发数
        self._batch_monitor_executor = ThreadPoolExecutor(
            max_workers=max_monitor_workers,
            thread_name_prefix='BatchMonitorThread'
        )
        self.logger.info("批处理相关属性已初始化。")

        self.logger.info("WQB_API_Client 初始化完成。")

    def _initialize_session(self) -> requests.Session:
        """
        初始化并配置一个新的 requests.Session 对象。
        这包括设置通用的请求头部信息。

        Returns:
            requests.Session: 新创建并配置好的会话对象。
        """
        self.logger.debug("开始初始化新的 requests.Session 对象...")
        session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'AlphaGenClient/1.0'  # 自定义 User-Agent
        })
        self.logger.info("新的 requests.Session 对象已初始化并配置了通用头部。")
        return session

    def _load_session(self) -> requests.Session | None:
        """
        尝试从本地文件加载已保存的 requests.Session 对象。

        会检查会话文件是否存在以及是否在有效期内。
        如果加载成功且会话有效，则将加载的会话赋值给 self.session 并返回。

        Returns:
            requests.Session | None: 如果成功加载且会话有效，则返回加载的会话对象；
                                     否则返回 None。
        """
        self.logger.debug(f"尝试从 '{self.session_pkl_file}' 加载会话...")
        if not os.path.exists(self.session_pkl_file):
            self.logger.info(f"会话文件 '{self.session_pkl_file}' 不存在。无法加载会话。")
            return None

        try:
            # 检查会话文件最后修改时间
            mod_time_timestamp = os.path.getmtime(self.session_pkl_file)
            mod_time = datetime.fromtimestamp(mod_time_timestamp)
            max_age_hours = self.config_manager.get('general.session_max_age_hours', 4) # 默认4小时

            if datetime.now() > mod_time + timedelta(hours=max_age_hours):
                self.logger.warning(f"会话文件 '{self.session_pkl_file}' 已过期 (超过 {max_age_hours} 小时)。")
                os.remove(self.session_pkl_file) # 删除过期的会话文件
                self.logger.info(f"已删除过期的会话文件: '{self.session_pkl_file}'")
                return None

            with open(self.session_pkl_file, 'rb') as f:
                loaded_session = pickle.load(f)

            # 简单验证加载的是否为 Session 对象 (可选)
            if not isinstance(loaded_session, requests.Session):
                self.logger.warning(f"从 '{self.session_pkl_file}' 加载的对象不是有效的 Session 类型。")
                return None

            # self.session = loaded_session # 赋值操作应由调用者决定或在更高层逻辑中处理
            self.logger.info(f"会话已成功从 '{self.session_pkl_file}' 加载。")
            return loaded_session
        except FileNotFoundError: # 虽然上面有exists检查，但以防万一在检查和打开之间文件被删除
            self.logger.info(f"会话文件 '{self.session_pkl_file}' 在尝试打开时未找到。")
            return None
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e: # 常见的pickle错误
            self.logger.warning(f"加载或反序列化会话文件 '{self.session_pkl_file}' 失败: {e}。可能文件已损坏或格式不兼容。")
            # 可以考虑删除损坏的文件
            try:
                os.remove(self.session_pkl_file)
                self.logger.info(f"已删除可能已损坏的会话文件: '{self.session_pkl_file}'")
            except OSError as rm_err:
                self.logger.error(f"删除损坏的会话文件 '{self.session_pkl_file}' 时出错: {rm_err}")
            return None
        except OSError as e: # os.path.getmtime 可能的错误
            self.logger.error(f"检查会话文件 '{self.session_pkl_file}' 修改时间时出错: {e}")
            return None
        except Exception as e: # 其他未知异常
            self.logger.error(f"加载会话文件 '{self.session_pkl_file}' 时发生未知错误: {e}", exc_info=True)
            return None

    def _save_session(self) -> None:
        """
        将当前的 self.session 对象序列化并保存到本地文件。

        如果 self.session 为 None，则不会执行任何操作。
        """
        if self.session is None:
            self.logger.warning("当前会话 (self.session) 为 None，无法保存会话。")
            return

        self.logger.debug(f"尝试将会话保存到 '{self.session_pkl_file}'...")
        try:
            # 确保目标目录存在
            session_dir = os.path.dirname(self.session_pkl_file)
            if not os.path.exists(session_dir):
                os.makedirs(session_dir, exist_ok=True)
                self.logger.info(f"会话目录 '{session_dir}' 不存在，已创建。")

            with open(self.session_pkl_file, 'wb') as f:
                pickle.dump(self.session, f)
            self.logger.info(f"会话已成功保存到 '{self.session_pkl_file}'。")
        except pickle.PicklingError as e:
            self.logger.error(f"序列化会话到 '{self.session_pkl_file}' 失败: {e}")
        except OSError as e: # os.makedirs 或文件打开/写入错误
            self.logger.error(f"保存会话到 '{self.session_pkl_file}' 时发生文件系统错误: {e}")
        except Exception as e: # 其他未知异常
            self.logger.error(f"保存会话到 '{self.session_pkl_file}' 时发生未知错误: {e}", exc_info=True)

    def _load_credentials(self) -> tuple[str, str]:
        """
        从指定的JSON文件中加载用户凭证（邮箱和密码）。

        该方法会严格处理文件不存在、文件格式错误或关键字段缺失的情况，
        因为有效的凭证对于客户端后续操作至关重要。

        Returns:
            tuple[str, str]: 一个包含邮箱和密码的元组 (email, password)。

        Raises:
            FileNotFoundError: 如果凭证文件未找到。
            json.JSONDecodeError: 如果凭证文件内容不是有效的JSON。
            KeyError: 如果JSON对象中缺少 'email' 或 'password' 键。
            Exception: 其他在文件读取或处理中发生的未知异常。
        """
        self.logger.debug(f"尝试从 '{self.credentials_file}' 加载凭证...")
        try:
            with open(self.credentials_file, 'r', encoding='utf-8') as f:
                credentials_data = json.load(f)

            email = credentials_data['email']
            password = credentials_data['password']

            self.logger.info(f"凭证已成功从 '{self.credentials_file}' 加载。")
            return email, password

        except FileNotFoundError:
            self.logger.critical(f"凭证文件 '{self.credentials_file}' 未找到。系统无法在没有凭证的情况下继续运行。")
            raise # 重新抛出 FileNotFoundError，调用者需处理此关键错误
        except json.JSONDecodeError as e:
            self.logger.error(f"解析凭证文件 '{self.credentials_file}' 失败: JSON格式无效 - {e}")
            raise # 重新抛出 JSONDecodeError
        except KeyError as e:
            self.logger.error(f"凭证文件 '{self.credentials_file}' 中缺少必要的键: {e} (预期 'email' 和 'password')")
            raise # 重新抛出 KeyError
        except Exception as e:
            self.logger.error(f"加载凭证文件 '{self.credentials_file}' 时发生未知错误: {e}", exc_info=True)
            raise # 重新抛出其他未知异常

    def login(self) -> None:
        """
        执行登录到 WQB API 的操作。

        该方法首先加载凭证，然后尝试通过 POST 请求到登录端点进行认证。
        它特别处理了基于 Persona 的生物识别认证流程。
        成功登录后，会话将被保存以供后续使用。

        Raises:
            FileNotFoundError: 如果凭证文件未找到 (由 _load_credentials 抛出)。
            json.JSONDecodeError: 如果凭证文件格式无效 (由 _load_credentials 抛出)。
            KeyError: 如果凭证文件中缺少 'email' 或 'password' (由 _load_credentials 抛出)。
            Exception: 如果登录失败或发生其他未处理的错误。
        """
        self.logger.info("尝试登录 WQB API...")

        # 确保我们有一个基础会话对象
        if self.session is None:
            self.session = self._initialize_session()

        try:
            email, password = self._load_credentials()
            # 设置 HTTP Basic Auth 凭证，这将应用于此会话的所有后续请求
            self.session.auth = (email, password)
            self.logger.info("凭证已加载并设置为会话的基本认证。")

            # 初始登录请求
            # 对于 Basic Auth，通常不需要请求体，服务器从 Authorization 头部读取凭证
            # 如果API要求，可以在此添加 json={'email': email, 'other_params': 'value'}
            self.logger.debug(f"向登录端点 {self.login_endpoint} 发送初始POST请求...")
            r = self.session.post(self.login_endpoint)
            self.logger.debug(f"初始登录请求完成，状态码: {r.status_code}")

            # 处理 Persona 生物识别认证
            if r.status_code == 401 and r.headers.get('WWW-Authenticate') == 'persona':
                persona_location = r.headers.get('Location')
                if not persona_location:
                    self.logger.error("Persona认证请求，但响应中缺少 Location 头部。")
                    raise Exception("WQB Persona 认证错误：缺少 Location 头部。")

                # persona_url = urljoin(self.base_url, persona_location) # Location可能已是完整URL
                # WQB API 通常返回完整的 Location URL for Persona
                persona_url = persona_location
                self.logger.info(f"需要 Persona 生物识别认证。认证URL: {persona_url}")
                print(f"\n请在浏览器中打开以下URL完成生物识别认证: {persona_url}")

                while True: # 循环直到 Persona 认证成功或用户中断
                    try:
                        input("完成生物识别认证后，请按 Enter 键继续...")
                    except KeyboardInterrupt:
                        self.logger.warning("用户中断了 Persona 认证流程。")
                        raise Exception("WQB 登录被用户中断。")

                    self.logger.info(f"向 Persona URL {persona_url} 发送确认POST请求...")
                    # Persona 确认请求通常不需要 Basic Auth，但 session 默认会发送
                    # 如果 Persona 服务器对此敏感，应使用 self.session.post(persona_url, auth=None)
                    # 或创建一个新的临时 session。假设当前API服务器不介意额外的Auth头。
                    persona_r = self.session.post(persona_url) # Persona URL 本身作为端点
                    self.logger.debug(f"Persona 确认请求完成，状态码: {persona_r.status_code}")

                    if persona_r.status_code == 201: # Persona 认证成功
                        r = persona_r # 更新主响应对象为 Persona 响应
                        break
                    elif persona_r.status_code == 202: # Accepted, 但可能还在处理中
                        self.logger.info("Persona 认证仍在处理中 (状态码 202)，请稍后重试或确保认证已在浏览器端完成。")
                        print("提示：Persona 认证仍在处理中。请确保您已在浏览器中完成所有步骤。")
                        # 不中断循环，让用户决定何时按Enter重试
                    else:
                        self.logger.warning(f"Persona 认证尝试失败，状态码: {persona_r.status_code}。响应: {persona_r.text}")
                        print(f"Persona 认证似乎未成功 (状态码: {persona_r.status_code})。请确保您已在浏览器中完成认证，然后按 Enter 重试。")
                        # 不中断循环，让用户决定何时按Enter重试

            # 处理最终登录结果
            if r.status_code == 201: # 201 Created 通常表示资源创建成功，在这里代表会话创建/登录成功
                self.logger.info("WQB API 登录成功！")
                self._save_session()
                # 登录成功后，通常 Basic Auth 就不再需要了，因为会话cookie已建立
                # 可以选择清除它，以避免在后续请求中意外发送（除非API总是需要它）
                # self.session.auth = None
            else:
                log_message = f"WQB API 登录失败。状态码: {r.status_code}, 响应: {r.text}"
                self.logger.error(log_message)
                if r.status_code == 401: # 用户名/密码错误 (非Persona场景)
                    self.logger.warning(f"登录凭证可能不正确。请检查或删除凭证文件 '{self.credentials_file}' 以便下次重新输入。")
                # 登录失败后清除可能已设置的 (错误) auth，以防影响后续操作或重试
                self.session.auth = None
                raise Exception(f"WQB 登录失败。状态码: {r.status_code}。请检查凭证或日志。")

        except Exception as e:
            # 确保即使发生异常，auth 也被清除 (如果已设置)
            if hasattr(self.session, 'auth'):
                 self.session.auth = None
            self.logger.error(f"登录过程中发生异常: {e}", exc_info=True)
            # 重新抛出原始异常或包装后的异常
            if not isinstance(e, (FileNotFoundError, json.JSONDecodeError, KeyError)):
                 raise Exception(f"WQB 登录过程中发生未处理的错误: {e}")
            else:
                 raise # 重新抛出由 _load_credentials 引发的特定异常

    def add_to_blacklist(self, op_name: str, reason: str = "API_ERROR") -> None:
        """
        将指定的操作符或字段名添加到不可访问操作的黑名单中。

        如果同名的条目已存在于黑名单中（仅比较名称），则不会重复添加。
        更新后的黑名单会通过 ConfigManager 持久化。

        Args:
            op_name (str): 要加入黑名单的操作符或字段的名称。
            reason (str, optional): 加入黑名单的原因。默认为 "API_ERROR"。
        """
        # 检查 op_name 是否已在黑名单中 (仅比较 'name' 键)
        # self._inaccessible_ops_blacklist 存储的是字典列表
        if not any(item.get('name') == op_name for item in self._inaccessible_ops_blacklist if isinstance(item, dict)):
            timestamp = datetime.now().isoformat()
            blacklist_entry = {
                'name': op_name,
                'reason': reason,
                'timestamp': timestamp
            }
            self._inaccessible_ops_blacklist.append(blacklist_entry)

            try:
                self.config_manager.set_blacklist(self._inaccessible_ops_blacklist)
                self.logger.info(f"条目 '{op_name}' 已因原因 '{reason}' 加入操作符黑名单并已持久化。")
            except Exception as e:
                self.logger.error(f"将条目 '{op_name}' 加入黑名单后，持久化失败: {e}", exc_info=True)
                # 考虑是否回滚内存中的添加，或者标记为未持久化
                # 目前，它仍保留在内存中的黑名单里
        else:
            self.logger.debug(f"尝试将条目 '{op_name}' 加入黑名单，但它已存在。")

    def clear_blacklist(self) -> None:
        """
        清空内存中和持久化的操作符黑名单。
        """
        self.logger.info("开始清空操作符黑名单...")
        self._inaccessible_ops_blacklist = []
        try:
            self.config_manager.set_blacklist([])
            self.logger.info("操作符黑名单已在内存中清空并已成功持久化（清空文件）。")
        except Exception as e:
            self.logger.error(f"清空操作符黑名单后，持久化（清空文件）失败: {e}", exc_info=True)
            # 内存中的黑名单已清空，但持久化的可能未清空

    def _extract_and_add_to_blacklist(self, error_message: str) -> None:
        """
        尝试从API错误消息中提取操作符或字段名，并将其加入黑名单。

        此方法使用预定义的正则表达式模式进行匹配。
        注意：这些正则表达式是示例性的，可能需要根据实际的API错误消息格式进行调整。

        Args:
            error_message (str): 从API接收到的错误消息字符串。
        """
        if not error_message or not isinstance(error_message, str):
            self.logger.debug("提供的错误消息为空或非字符串，无法提取黑名单条目。")
            return

        self.logger.debug(f"尝试从错误消息中提取黑名单条目: '{error_message[:200]}...'") # Log a snippet

        # 示例正则表达式模式
        # 模式1: 匹配 "operator "OPERATOR_NAME" is not supported"
        pattern_op_not_supported = re.search(r'operator "([^"]+)" is not supported', error_message, re.IGNORECASE)
        if pattern_op_not_supported:
            extracted_name = pattern_op_not_supported.group(1)
            self.logger.info(f"从错误消息中提取到不支持的操作符: '{extracted_name}'。")
            self.add_to_blacklist(extracted_name, "EXTRACTED_OPERATOR_NOT_SUPPORTED")
            return # 通常一个错误消息只对应一个主要问题实体

        # 模式2: 匹配 "field "FIELD_NAME" is not accessible"
        pattern_field_not_accessible = re.search(r'field "([^"]+)" is not accessible', error_message, re.IGNORECASE)
        if pattern_field_not_accessible:
            extracted_name = pattern_field_not_accessible.group(1)
            self.logger.info(f"从错误消息中提取到不可访问的字段: '{extracted_name}'。")
            self.add_to_blacklist(extracted_name, "EXTRACTED_FIELD_NOT_ACCESSIBLE")
            return

        # 模式3: 匹配 "invalid field: FIELD_NAME" (假设的另一种格式)
        pattern_invalid_field = re.search(r'invalid field: (\w+)', error_message, re.IGNORECASE)
        if pattern_invalid_field:
            extracted_name = pattern_invalid_field.group(1)
            self.logger.info(f"从错误消息中提取到无效字段: '{extracted_name}'。")
            self.add_to_blacklist(extracted_name, "EXTRACTED_INVALID_FIELD")
            return

        self.logger.debug("未从错误消息中匹配到已知的黑名单条目模式。")

    def get_available_datafields(self) -> list:
        """
        从WQB API获取可用的数据集列表。
        
        该方法会向WQB API的data-sets端点发送GET请求，获取当前可用的数据集列表。
        成功获取后，会通过ConfigManager将数据集列表持久化保存。
        
        Returns:
            list: 可用数据集ID的列表。如果获取失败，返回空列表。
            
        Raises:
            Exception: 如果API请求失败或响应格式不正确。
        """
        self.logger.info("开始从WQB API获取可用数据集列表...")
        
        try:
            # 构建查询参数 - 根据WQB API要求添加必需参数
            params = {
                'delay': 1,
                'instrumentType': 'EQUITY',
                'limit': 20,
                'offset': 0,
                'region': 'USA',
                'universe': 'TOP3000'
            }
            
            # 从配置中获取自定义参数（如果存在）
            delay = self.config_manager.get('general.default_delay', 1)
            instrument_type = self.config_manager.get('general.default_instrument_type', 'EQUITY')
            region = self.config_manager.get('general.default_region', 'USA')
            universe = self.config_manager.get('general.default_universe', 'TOP3000')
            
            params.update({
                'delay': delay,
                'instrumentType': instrument_type,
                'region': region,
                'universe': universe
            })
            
            self.logger.debug(f"使用查询参数: {params}")
            
            # 收集所有数据集
            all_datasets = []
            
            while True:
                # 发送GET请求到data-sets端点
                response = self._send_request_with_retry(
                    method='GET',
                    url=self.data_sets_endpoint,
                    params=params
                )
                
                # 解析响应JSON
                try:
                    response_data = response.json()
                    self.logger.debug(f"收到数据集API响应 (offset={params['offset']}): 数据项数量={len(response_data.get('results', []))}. 响应内容片段: {response.text[:100]}...")
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析数据集API响应时发生JSON错误: {e}. 响应内容: {response.text[:500]}...")
                    # 如果JSON解析失败，则认为没有更多有效数据，中断循环
                    break
                
                # 从 'results' 字段获取数据集列表
                current_items = response_data.get('results', [])
                total_count = response_data.get('count', 0)  # 获取总数

                self.logger.debug(f"当前页数据项数量: {len(current_items)}, 总数据项数量: {total_count}, 每页限制: {params['limit']}")

                # 终止条件1: 如果当前页没有数据，则停止
                if not current_items:
                    self.logger.info("没有更多数据集，分页获取完成。")
                    break
                
                # 处理当前页的数据集，过滤掉id为"univ1"的数据集
                for item in current_items:
                    dataset_id = item.get('id')
                    if dataset_id and dataset_id != 'univ1':  # 舍弃id为"univ1"的数据集
                        all_datasets.append(dataset_id)
                
                # 终止条件2: 如果已获取的数据量达到或超过API报告的总数，则停止
                if total_count > 0 and len(all_datasets) >= total_count - 1:  # 减1是因为要舍弃univ1
                    self.logger.info(f"已获取所有可用数据集，分页获取完成。")
                    break

                # 终止条件3: 如果当前页数据少于限制数量，说明是最后一页，则停止
                if len(current_items) < params['limit']:
                    self.logger.info("当前页数据少于限制数量，已获取所有数据集。")
                    break
                
                # 更新offset以获取下一页
                params['offset'] += params['limit']
                self.logger.debug(f"准备获取下一页数据，新offset: {params['offset']}")
            
            # 去重并排序
            unique_datasets = sorted(list(set(all_datasets)))
            
            self.logger.info(f"成功获取到 {len(unique_datasets)} 个唯一数据集")
            
            # 通过ConfigManager保存数据集列表
            try:
                self.config_manager.set_datasets(unique_datasets)
                self.logger.info("数据集列表已成功保存到配置中")
            except Exception as save_error:
                self.logger.error(f"保存数据集列表时发生错误: {save_error}", exc_info=True)
                # 即使保存失败，仍然返回获取到的数据
            
            return unique_datasets
            
        except requests.exceptions.RequestException as req_error:
            self.logger.error(f"获取数据集列表时发生网络错误: {req_error}", exc_info=True)
            raise Exception(f"网络请求失败: {req_error}")
        except json.JSONDecodeError as json_error:
            self.logger.error(f"解析数据集API响应时发生JSON错误: {json_error}", exc_info=True)
            raise Exception(f"API响应格式错误: {json_error}")
        except Exception as e:
            self.logger.error(f"获取数据集列表时发生未知错误: {e}", exc_info=True)
            raise Exception(f"获取数据集失败: {e}")


    def _send_request_with_retry(self, method: str, url: str,
                                 json_data: dict = None, params: dict = None,
                                 max_retries: int = None) -> requests.Response:
        """
        发送带有重试逻辑的HTTP请求。

        该方法会自动处理会话刷新、API限流(429)、认证失败(401)以及网络相关的临时错误。
        对于不可恢复的错误（如403 Forbidden），则不会重试。

        Args:
            method (str): HTTP方法 (例如 'GET', 'POST', 'PUT', 'DELETE')。
            url (str): 请求的目标URL。
            json_data (dict, optional): 对于POST/PUT等请求，要作为JSON发送的数据。默认为None。
            params (dict, optional): URL查询参数。默认为None。
            max_retries (int, optional): 最大重试次数。如果为None，则从配置中获取
                                         ('general.api_max_retries', 默认3次)。

        Returns:
            requests.Response: 成功响应的requests.Response对象。

        Raises:
            requests.exceptions.HTTPError: 如果在多次重试后仍收到4xx/5xx错误 (401, 429除外，它们有特定处理)，
                                           或者收到403错误。
            requests.exceptions.RequestException: 如果在多次重试后仍发生网络连接错误或超时。
            Exception: 如果所有重试尝试均告失败，或者发生其他未处理的严重错误。
        """
        if max_retries is None:
            max_retries = self.config_manager.get('general.api_max_retries', 3)

        # 确保会话是有效的 (这可能会触发登录)
        self.refresh_session_if_needed()

        self.logger.debug(f"准备发送请求 (最多 {max_retries + 1} 次尝试): {method} {url}")
        self.logger.debug(f"请求参数 (params): {params}")
        self.logger.debug(f"请求体 (json_data): {json_data if json_data else '无'}")


        for attempt in range(max_retries + 1):
            self.logger.info(f"发送请求: {method} {url} (尝试 {attempt + 1}/{max_retries + 1})")
            try:
                response = self.session.request(
                    method,
                    url,
                    json=json_data,
                    params=params,
                    timeout=self.config_manager.get('general.api_timeout_seconds', 60)
                )
                self.logger.debug(f"收到响应: {response.status_code} {response.reason} from {url}")

                if response.ok: # 2xx 状态码
                    return response

                # --- 特定状态码处理 ---
                if response.status_code == 401: # Unauthorized
                    self.logger.warning(f"请求 {url} 返回401 Unauthorized (尝试 {attempt + 1})。会话可能已过期或无效。将尝试重新登录。")
                    if attempt < max_retries:
                        self.login() # 尝试重新登录，这会更新会话
                        self.logger.info("重新登录完成，将在短暂延迟后重试原始请求。")
                        time.sleep(1) # 给登录过程一点时间，并避免立即冲击服务器
                        continue # 继续下一次重试迭代
                    else:
                        self.logger.error(f"即使在重新登录后，请求 {url} 仍返回401 (在所有重试后)。")
                        # 抛出原始响应的HTTPError，以便调用者可以检查
                        response.raise_for_status() # 这将引发一个HTTPError

                elif response.status_code == 403: # Forbidden
                    self.logger.critical(
                        f"请求 {url} 返回403 Forbidden。这通常表示权限不足，不会重试。"
                        f"请求数据: {json_data if json_data else params if params else '无'}"
                    )
                    # TODO: 未来可以考虑调用 self.add_to_blacklist("SUSPECTED_FORBIDDEN_RESOURCE", f"403_error_at_{url}")
                    response.raise_for_status() # 抛出HTTPError，不再重试

                elif response.status_code == 429: # Too Many Requests
                    retry_after_str = response.headers.get('Retry-After')
                    if retry_after_str and retry_after_str.isdigit():
                        wait_seconds = int(retry_after_str)
                    else:
                        wait_seconds = self.config_manager.get('general.api_retry_after_default_seconds', 5)

                    self.logger.warning(
                        f"请求 {url} 返回429 Too Many Requests (尝试 {attempt + 1})。"
                        f"将根据 Retry-After 头部或默认值等待 {wait_seconds} 秒。"
                    )
                    if attempt < max_retries:
                        time.sleep(wait_seconds)
                        continue
                    else:
                        self.logger.error(f"即使在等待后，请求 {url} 仍返回429 (在所有重试后)。")
                        response.raise_for_status()


                # --- 其他 4xx 或 5xx 错误 ---
                # 对于其他客户端错误 (4xx) 或服务器错误 (5xx)，进行记录和可能的指数退避重试
                self.logger.error(
                    f"请求 {url} 失败 (尝试 {attempt + 1})，状态码: {response.status_code}。"
                    f"响应: {response.text}" # 记录完整响应文本以供调试
                )
                if attempt == max_retries:
                    self.logger.error(f"已达到最大重试次数 ({max_retries + 1})，请求 {url} 最终失败。")
                    response.raise_for_status() # 抛出最终的HTTPError

                # 执行指数退避
                backoff_factor = self.config_manager.get('general.api_retry_backoff_factor', 1)
                max_backoff = self.config_manager.get('general.api_retry_max_backoff_seconds', 30)
                delay = min(backoff_factor * (2 ** attempt), max_backoff)
                self.logger.info(f"将等待 {delay:.2f} 秒后重试请求 {url}...")
                time.sleep(delay)
                continue # 继续下一次迭代

            except requests.exceptions.ConnectionError as e:
                error_msg = f"请求 {url} 发生连接错误 (尝试 {attempt + 1}): {e}"
                self.logger.warning(error_msg)
                if attempt == max_retries:
                    self.logger.error(f"已达到最大重试次数，连接错误导致请求 {url} 最终失败。")
                    raise
            except requests.exceptions.Timeout as e:
                error_msg = f"请求 {url} 超时 (尝试 {attempt + 1}): {e}"
                self.logger.warning(error_msg)
                if attempt == max_retries:
                    self.logger.error(f"已达到最大重试次数，超时导致请求 {url} 最终失败。")
                    raise
            except requests.exceptions.RequestException as e: # 其他requests库相关的通用异常
                error_msg = f"请求 {url} 发生 RequestException (尝试 {attempt + 1}): {e}"
                self.logger.error(error_msg, exc_info=True) # 记录堆栈跟踪
                if attempt == max_retries:
                    self.logger.error(f"已达到最大重试次数，RequestException导致请求 {url} 最终失败。")
                    raise

            # 如果是网络相关的异常并且不是最后一次尝试，则执行指数退避
            if attempt < max_retries: # 这个判断确保了上面raise之后不会执行下面的退避
                backoff_factor = self.config_manager.get('general.api_retry_backoff_factor', 1)
                max_backoff = self.config_manager.get('general.api_retry_max_backoff_seconds', 30)
                delay = min(backoff_factor * (2 ** attempt), max_backoff)
                self.logger.info(f"将等待 {delay:.2f} 秒后重试因网络问题的请求 {url}...")
                time.sleep(delay)

        # 如果循环结束仍未成功返回响应
        final_error_msg = f"API请求在 {max_retries + 1} 次尝试后最终失败: {method} {url}"
        self.logger.critical(final_error_msg)
        raise Exception(final_error_msg)

    def refresh_session_if_needed(self) -> None:
        """
        智能地检查并刷新当前WQB API会话，确保其有效性。

        处理流程优先级如下：
        1. 如果当前没有有效凭证 (`self.session.auth` 未设置)，则执行完整登录。
        2. 尝试从本地文件加载并验证已保存的会话 (`_load_session`)。
        3. 如果无法从文件加载有效会话，则尝试使用当前内存中的凭证“刷新”会话（通过再次POST到登录端点）。
        4. 如果“刷新”失败或凭证无效，则执行完整的重新登录流程 (`self.login()`)。
        5. 任何步骤中发生严重错误，则尝试回退到完整登录。

        此方法旨在被其他需要认证的API调用前调用，以确保会话尽可能保持活动状态。
        """
        self.logger.info("开始检查并刷新WQB API会话（如果需要）...")

        try:
            # 1. 检查 self.session 和 self.session.auth 是否存在
            # self.session 应该总是在 __init__ 中被初始化
            if self.session is None: # 理论上不应发生，除非 __init__ 失败或被外部修改
                self.logger.warning("Session对象 (self.session) 未初始化。将尝试执行完整登录。")
                self.login()
                return

            if not hasattr(self.session, 'auth') or self.session.auth is None:
                self.logger.info("当前会话没有设置认证凭证 (self.session.auth)。将执行完整登录。")
                self.login()
                return

            # 2. 尝试从文件加载会话 (包含了有效期检查)
            self.logger.debug("尝试从文件加载并验证会话...")
            loaded_session = self._load_session()
            if loaded_session:
                self.session = loaded_session # _load_session 成功加载并验证了会话
                self.logger.info("会话已成功从文件恢复并验证有效。")
                # 确保 auth 信息也从加载的会话中恢复 (pickle 应该处理了 session.auth)
                # 如果需要，可以额外检查 self.session.auth 是否存在
                return

            # 3. 如果无法从文件加载，尝试使用当前凭证“刷新”会话
            self.logger.info("未能从文件加载有效会话。尝试使用当前凭证刷新会话...")

            # 确保凭证仍然加载在 session.auth 中，如果之前登录失败可能被清除了
            # 或者 _load_credentials 失败了，这里会再次尝试
            try:
                if not self.session.auth: # 检查 auth 是否为 None 或空元组
                    self.logger.debug("session.auth 为空，尝试重新加载凭证以进行刷新操作。")
                    email, password = self._load_credentials()
                    self.session.auth = (email, password)
            except Exception as cred_err: # _load_credentials 可能抛出异常
                self.logger.error(f"刷新会话前加载凭证失败: {cred_err}。将尝试完整登录。")
                self.login()
                return

            self.logger.debug(f"向登录端点 {self.login_endpoint} 发送POST请求以刷新会话...")
            r = self.session.post(self.login_endpoint) # 使用已设置的 self.session.auth

            if r.status_code == 201: # 刷新成功
                self.logger.info("会话通过再次认证刷新成功。")
                self._save_session() # 保存刷新后的会话 (可能包含新的cookie等)
                return
            elif r.status_code == 401: # 凭证无效或会话已过期
                self.logger.warning("使用当前凭证刷新会话失败 (401 Unauthorized)。凭证可能已过期或无效。将执行完整登录。")
                self.login() # 执行完整登录流程，这会处理Persona等情况
                return
            else: # 其他非预期错误
                self.logger.error(f"会话刷新尝试失败，状态码: {r.status_code}, 响应: {r.text}。将尝试完整登录。")
                self.login()
                return

        except Exception as e:
            self.logger.error(f"刷新会话过程中发生未知错误: {e}。将尝试执行完整登录作为最终手段。", exc_info=True)
            try:
                self.login()
            except Exception as login_e:
                self.logger.critical(f"最终尝试的完整登录也失败了: {login_e}", exc_info=True)
                # 此处异常会自然向上传播，或者可以抛出特定的会话管理异常
                raise # 重新抛出登录时发生的异常

    def get_dataset_datafields(self, dataset_id: str) -> list:
        """
        获取指定数据集的数据字段列表。

        Args:
            dataset_id (str): 数据集ID。

        Returns:
            list: 数据字段列表，如果获取失败则返回空列表。
        """
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            self.logger.error("get_dataset_datafields 接收到的数据集ID无效。")
            return []

        self.logger.info(f"开始获取数据集 {dataset_id} 的数据字段列表...")
        
        # 确保会话有效
        self.refresh_session_if_needed()
        
        all_datafields = []
        offset = 0
        limit = 50
        
        # 构建基础查询参数
        base_params = {
            'dataset.id': dataset_id,
            'delay': 1,
            'instrumentType': 'EQUITY',
            'limit': limit,
            'region': 'USA',
            'universe': 'TOP3000'
        }
        
        # 从配置中获取自定义参数（如果有）
        custom_params = self.config_manager.get('api.datafields_params', {})
        if custom_params:
            base_params.update(custom_params)
            self.logger.debug(f"应用自定义数据字段查询参数: {custom_params}")
        
        try:
            while True:
                # 设置当前页的offset
                params = base_params.copy()
                params['offset'] = offset
                
                self.logger.debug(f"请求数据集 {dataset_id} 的数据字段，offset={offset}, limit={limit}")
                
                # 发送GET请求
                response = self._send_request_with_retry(
                    method='GET',
                    url=self.data_fields_endpoint,
                    params=params
                )
                
                if not response:
                    self.logger.error(f"获取数据集 {dataset_id} 的数据字段时请求失败。")
                    break
                
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析数据集 {dataset_id} 的数据字段响应JSON失败: {e}")
                    break
                
                # 提取数据字段列表
                current_datafields = []
                if 'results' in data and isinstance(data['results'], list):
                    current_datafields = data['results']
                elif 'data' in data and isinstance(data['data'], list):
                    current_datafields = data['data']
                else:
                    self.logger.warning(f"数据集 {dataset_id} 的响应中未找到 'results' 或 'data' 字段，或字段不是列表类型。")
                    break
                
                if not current_datafields:
                    self.logger.info(f"数据集 {dataset_id} 在offset={offset}处没有更多数据字段，停止分页。")
                    break
                
                all_datafields.extend(current_datafields)
                self.logger.debug(f"数据集 {dataset_id} 当前页获取到 {len(current_datafields)} 个数据字段")
                
                # 检查是否需要继续分页
                total_count = data.get('count', 0)
                if total_count > 0 and len(all_datafields) >= total_count:
                    self.logger.info(f"数据集 {dataset_id} 已获取所有数据字段 (总数: {total_count})")
                    break
                
                if len(current_datafields) < limit:
                    self.logger.info(f"数据集 {dataset_id} 当前页数据字段数量 ({len(current_datafields)}) 少于限制 ({limit})，停止分页。")
                    break
                
                offset += limit
                
                # 防止无限循环的安全检查
                if offset > 10000:  # 假设单个数据集不会超过10000个字段
                    self.logger.warning(f"数据集 {dataset_id} 的数据字段获取已达到安全限制 (offset > 10000)，停止分页。")
                    break
        
        except Exception as e:
            self.logger.error(f"获取数据集 {dataset_id} 的数据字段时发生异常: {e}", exc_info=True)
            return []
        
        # 去重和排序
        unique_datafields = []
        seen_ids = set()
        for field in all_datafields:
            if isinstance(field, dict) and 'id' in field:
                field_id = field['id']
                if field_id not in seen_ids:
                    seen_ids.add(field_id)
                    unique_datafields.append(field)
        
        # 按字段ID排序
        unique_datafields.sort(key=lambda x: x.get('id', ''))
        
        self.logger.info(f"数据集 {dataset_id} 共获取到 {len(unique_datafields)} 个唯一数据字段")
        
        # 保存到配置管理器
        try:
            self.config_manager.set_dataset_datafields(dataset_id, unique_datafields)
        except Exception as e:
            self.logger.error(f"保存数据集 {dataset_id} 的数据字段到配置管理器失败: {e}")
        
        return unique_datafields

    def get_all_datasets_datafields(self) -> dict:
        """
        获取所有数据集的数据字段列表。

        Returns:
            dict: 以数据集ID为键，数据字段列表为值的字典。
        """
        self.logger.info("开始获取所有数据集的数据字段列表...")
        
        # 首先获取数据集列表
        datasets = self.get_available_datafields()  # 这个方法现在返回数据集ID字符串列表
        if not datasets:
            self.logger.warning("未获取到任何数据集，无法获取数据字段。")
            return {}
        
        self.logger.info(f"获取到 {len(datasets)} 个数据集，开始逐个获取数据字段...")
        all_dataset_fields = {}
        
        for dataset_id in datasets:
            # 数据集现在是字符串ID，不再是包含'id'字段的对象
            if isinstance(dataset_id, str) and dataset_id.strip():
                self.logger.info(f"正在获取数据集 {dataset_id} 的数据字段...")
                
                datafields = self.get_dataset_datafields(dataset_id)
                if datafields:
                    all_dataset_fields[dataset_id] = datafields
                    self.logger.info(f"数据集 {dataset_id} 获取到 {len(datafields)} 个数据字段")
                else:
                    self.logger.warning(f"数据集 {dataset_id} 未获取到任何数据字段")
            else:
                self.logger.warning(f"跳过无效的数据集ID: {dataset_id}")
        
        self.logger.info(f"所有数据集数据字段获取完成，共处理 {len(all_dataset_fields)} 个数据集")
        return all_dataset_fields

# 主程序入口示例 (用于基本测试或演示)
if __name__ == '__main__':
    # 假设 ConfigManager 已存在且可以被实例化
    # 需要创建一个模拟的 ConfigManager 或确保 ConfigManager 已实现并可导入
    class MockConfigManager:
        def __init__(self):
            self.config_data = {
                "general": {
                    "api_base_url": "https://testapi.example.com/",
                    "credentials_file": "config/test_creds.json",
                    "session_pkl_file": "data/test_session.pkl",
                    "max_concurrent_simulations": 3
                },
                "blacklist": ["bad_op1", "bad_op2"]
            }
            self.logger = logging.getLogger(__name__) # Mock CM 也有 logger

        def get(self, key_path: str, default=None):
            keys = key_path.split('.')
            val = self.config_data
            try:
                for key in keys:
                    val = val[key]
                return val
            except KeyError:
                self.logger.warning(f"MockConfigManager: Key '{key_path}' not found, returning default '{default}'")
                return default
            except TypeError: # Handle cases where an intermediate key is not a dict
                 self.logger.warning(f"MockConfigManager: Path '{key_path}' is invalid, an intermediate key is not a dictionary. Returning default '{default}'")
                 return default


    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    mock_cm = MockConfigManager()

    try:
        client = WQB_API_Client(config_manager=mock_cm)
        # 可以在这里打印一些client的属性来验证
        logging.info(f"Client Base URL: {client.base_url}")
        logging.info(f"Client Login Endpoint: {client.login_endpoint}")
        logging.info(f"Client Inaccessible Ops: {client._inaccessible_ops_blacklist}")
        logging.info(f"Client Session object: {client.session}")
    except Exception as e:
        logging.error(f"在 WQB_API_Client 初始化过程中发生错误: {e}", exc_info=True)

    # 清理执行器资源 (在实际应用中，这通常在程序关闭时完成)
    if 'client' in locals() and client:
        client._batch_submit_executor.shutdown(wait=False)
        client._batch_monitor_executor.shutdown(wait=False)
        logging.info("批处理执行器已关闭。")
