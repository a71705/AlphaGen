# src/utils.py
import logging
import os
from datetime import datetime

def setup_logging(base_log_dir="logs", log_level=logging.INFO):
    """
    配置系统日志，按日期创建文件夹。

    Args:
        base_log_dir (str): 基础日志目录的路径。
        log_level (int): 日志记录的级别 (例如 logging.INFO, logging.DEBUG)。

    Returns:
        logging.Logger: 配置好的根 logger 实例。
    """
    # 根据当前日期创建日志子目录路径
    # 例如 logs/YYYY-MM-DD/
    log_date_dir = os.path.join(base_log_dir, datetime.now().strftime("%Y-%m-%d"))
    # 创建日期目录，如果目录已存在则不报错
    os.makedirs(log_date_dir, exist_ok=True)

    # 定义系统日志文件和错误日志文件的完整路径
    system_log_path = os.path.join(log_date_dir, "system.log")
    errors_log_path = os.path.join(log_date_dir, "errors.log")

    # 获取根 logger 实例
    logger = logging.getLogger()
    # 设置 logger 的最低日志级别
    logger.setLevel(log_level)

    # 清除已存在的处理器，防止重复添加 (特别是在多次调用此函数或模块重载时)
    # 遍历 logger.handlers 的副本 (logger.handlers[:]) 进行操作，因为在迭代过程中修改列表是不安全的
    for handler in logger.handlers[:]:
        # 检查处理器是否是 FileHandler 或 StreamHandler 的实例
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            logger.removeHandler(handler) # 移除已存在的处理器

    # 创建文件处理器 (FileHandler) 用于记录所有 INFO 级别及以上的日志到 system.log
    # 使用 'utf-8' 编码以支持中文等非 ASCII 字符
    file_handler = logging.FileHandler(system_log_path, encoding='utf-8')
    file_handler.setLevel(log_level) # 设置此处理器的日志级别
    # 定义日志格式
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(file_formatter) # 将格式应用到处理器
    logger.addHandler(file_handler) # 将处理器添加到 logger

    # 创建文件处理器 (FileHandler) 专门用于记录 ERROR 级别及以上的日志到 errors.log
    error_file_handler = logging.FileHandler(errors_log_path, encoding='utf-8')
    error_file_handler.setLevel(logging.ERROR) # 只记录 ERROR 及更高级别的日志
    error_file_handler.setFormatter(file_formatter) # 使用与上面相同的日志格式
    logger.addHandler(error_file_handler) # 将错误日志处理器添加到 logger

    # 创建控制台处理器 (StreamHandler) 用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level) # 设置控制台输出的日志级别
    # 定义控制台输出的日志格式 (可以比文件格式更简洁)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter) # 应用格式
    logger.addHandler(console_handler) # 添加到 logger

    # 记录一条信息，表示日志系统已配置完成
    logging.info(f"日志已配置，输出到 {log_date_dir}")
    # 返回配置好的 logger 实例，方便其他模块直接使用或进一步配置
    return logger
