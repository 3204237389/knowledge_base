import asyncio
import json

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from tool.web_search_tool import WebSearchTool


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        logger.info(f"【{self.name}】节点逻辑")

        query = state.get("rewritten_query")
        docs = []

        result_dict = asyncio.new_event_loop().run_until_complete(self._mcp_call(query))

        pages = result_dict.get("pages", [])
        for page in pages:
            snippet = (page.get("snippet") or "").strip()
            url = (page.get("url") or "").strip()
            title = (page.get("title") or "").strip()
            docs.append({"snippet": snippet, "url": url, "title": title})

        return {"web_search_docs": docs}

    async def _mcp_call(self, query):
        tool = WebSearchTool(max_results=3)
        result = await tool.search(query)
        return result


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(result)
