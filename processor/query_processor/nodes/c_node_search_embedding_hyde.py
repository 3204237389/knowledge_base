import json

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        logger.info(f"【{self.name}】节点逻辑")
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")

        hyde_doc = self._step_1_create_embedding_hyde_doc(rewritten_query)
        print(hyde_doc)
        response = self._step_2_search_embedding_hyde(
            rewritten_query = rewritten_query,
            hyde_doc = hyde_doc,
            item_names = item_names
        )

        json_response = json.dumps(response,ensure_ascii=False,indent=4)

        return {"hyde_embedding_chunks": json_response,"hyde_doc":hyde_doc}

    def _step_1_create_embedding_hyde_doc(self, rewritten_query:str)->str:
        milvus_client = get_milvus_client()

        hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)

        ai_client = get_llm_client()
        hyde_doc = ai_client.invoke(hyde_prompt).content

        return hyde_doc

    def _step_2_search_embedding_hyde(self, rewritten_query:str, hyde_doc:str, item_names:list):
        embedding_context = rewritten_query + " " +hyde_doc

        embeddings = generate_embeddings([embedding_context])
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]

        milvus_client = get_milvus_client()
        collection_name = milvus_config.chunks_collection

        res = create_hybrid_search_requests(
            dense_vector=dense,
            sparse_vector=sparse,
            expr=f'item_name in {item_names}'
        )

        response = hybrid_search(
            client=milvus_client,
            collection_name=collection_name,
            reqs=res,
            limit=5,
            output_fields=["chunk_id", "content", "item_name"]
        )

        print(response[0])
        return {"embedding_chunks": response[0] if response else []}


if __name__ == "__main__":
    node_search_embedding = NodeSearchEmbeddingHyde()
    init_state = {
        "item_names" : ["兄弟HAK180烫金机","百度一下"],
        "rewritten_query": "兄弟帮我查询一下HAK180烫金机是什么？"
    }

    result = node_search_embedding.process(init_state)
