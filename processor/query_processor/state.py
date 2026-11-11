from typing import TypedDict, List

class QueryGraphState(TypedDict):
    """
    查询流程图状态
    包含整个查询流程中传递的所有数据。
    """

    session_id: str
    message_id: str

    original_query: str

    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list

    rrf_chunks: list
    reranked_docs: list

    prompt: str
    answer: str

    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
