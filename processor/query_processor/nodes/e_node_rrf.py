import json
from typing import List, Tuple

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        logger.info(f"【{self.name}】节点逻辑")

        embedding_chunks = state.get("embedding_chunks")
        hyde_embedding_chunks = state.get("hyde_embedding_chunks")
        embedding_chunks_list = [doc.get("entity") for doc in embedding_chunks]
        hyde_embedding_chunks_list = [doc.get("entity") for doc in hyde_embedding_chunks]

        rrf_inputs = [
            (embedding_chunks_list,1.0),
            (hyde_embedding_chunks_list,1.0)
        ]

        rrf_merge_results= self._rrf_merge(rrf_inputs)

        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        state["rrf_chunks"] = rrf_chunks
        return state

    def _rrf_merge(self, rrf_inputs:List[Tuple],k:int=60,max_results: int=5):
        print("向量搜索和假设性搜索的融合函数")
        chunk_scores = {}
        chunk_data = {}

        for rrf_inputs,weight in rrf_inputs:
            for rank,doc in enumerate(rrf_inputs):
                chunk_id = doc.get("chunk_id")
                chunk_scores[chunk_id]=chunk_scores.get(chunk_id,0.0)+weight/(k+rank)
                chunk_data.setdefault(chunk_id,doc)

        unsorted_results = [(chunk_data[cid],score) for cid,score in chunk_scores.items()]

        sorted_results = sorted(
            unsorted_results,
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:max_results] if max_results else sorted_results


if __name__ == "__main__":
    mock_state = {
            "embedding_chunks": [
                {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
                {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
                {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
            ],
            "hyde_embedding_chunks": [
                {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
                {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
                {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
            ]
        }

    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
    logger.info(json.dumps(result, indent=4))
