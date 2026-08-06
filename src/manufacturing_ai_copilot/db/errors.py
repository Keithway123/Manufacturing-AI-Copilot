class DatabaseError(Exception):
    """数据库层项目异常的基础类型"""


class DatabaseUnavailableError(DatabaseError):
    """数据库连接不可用，不表示SQL业务逻辑错误"""
