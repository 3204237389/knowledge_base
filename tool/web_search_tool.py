import json
import httpx

from config.tavily_config import tavily_config
from tool.logger import logger


class WebSearchTool:
    """
    联网搜索工具（异步版）：调用 Tavily 进行搜索
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.api_url = "https://api.tavily.com/search"

    async def search(self, query: str) -> dict:
        """
        执行异步联网搜索
        :param query: 搜索问题
        :return: {"pages": [{"title": "", "url": "", "snippet": ""}]}
        """
        logger.info(f"【WebSearchTool】开始异步搜索：{query}")

        payload = {
            "api_key": tavily_config.api_key,
            "query": query,
            "max_results": self.max_results,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()

        pages = []
        for item in data.get("results", []):
            pages.append({
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "snippet": (item.get("content") or "").strip()
            })

        return {"pages": pages}
