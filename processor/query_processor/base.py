"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能
"""
from abc import abstractmethod, ABC
from typing import TypeVar
from tool.logger import logger

T = TypeVar("T")
class NodeBase(ABC):

    name: str = "base_node"

    def __call__(self, state: T) -> T:
        """
        节点执行入口
        """
        try:
            logger.info(f"--- {self.name} 开始啦 ---")

            result = self.process(state)

            logger.info(f"--- {self.name} 完成啦 ---")

            return result

        except Exception as e:
            logger.error(f"{self.name} 执行失败: {e}")
            raise

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑
        子类必须实现此方法
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        pass
