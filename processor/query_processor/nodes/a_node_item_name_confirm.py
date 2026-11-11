import json
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, \
    ITEM_NAME_EXTRACT_TEMPLATE
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        session_id, original_query = self._step_1_validate_param(state)

        history = get_recent_messages(session_id)
        state["history"] = history

        message_id = save_chat_message(session_id, "user", original_query)

        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res["item_names"]
        rewritten_query = extract_res["rewritten_query"]
        state["item_names"] = item_names
        state["rewritten_query"] = rewritten_query

        align_result = {}
        if len(item_names) >= 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        state = self._step_7_check_confirmation(state, align_result, history)

        self._step_8_write_history(state, session_id, rewritten_query, message_id)

        logger.info(f"【{self.name}】节点逻辑")

        return state

    def _step_1_validate_param(self, state):
        print("step_1：参数校验")
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query

    def _step_4_extract_info(self, original_query: str, history: dict) -> Dict:
        print("step_4:模型提取意图主题")

        ai_client = get_llm_client()

        history_text = ""
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            history_text = f"{role}:{content}\n"

        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query
        )

        message = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = ai_client.invoke(message)
        response_content = response.content

        result = json.loads(response_content)
        print(result)
        if "item_names" not in result:
            result["item_names"] = []
        if "original_query" not in result:
            result["original_query"] = original_query

        if result["item_names"]:
            result["item_names"] = [name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                                    for name in result["item_names"]]

        return result

    def _step_5_vectorize_and_query(self, item_names: list) -> List[Dict]:
        print("step_5：向量检索,检索出对应的item_name的相似的商品名称列表")
        results: List[Dict] = []

        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection

        embeddings = generate_embeddings(item_names)

        for i in range(len(item_names)):
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]
            reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector, limit=5)
            search_res = hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8, 0.2),
                norm_score=True,
                output_fields=["item_name"]
            )

            matches = []
            if search_res and len(search_res) > 0:
                print("此处是意图识别的向量搜索结果：")
                print(search_res)
                for hit in search_res[0]:
                    matches.append({
                        "item_name": hit.entity.get("item_name"),
                        "score": hit.get("distance")
                    })

            results.append({
                "extracted_name": item_names[i],
                "matches": matches
            })

        return results

    def _step_6_align_item_names(self, query_results) -> Dict:
        print("step_6：对齐结果，高分和低分结果整合")
        confirmed_item_names: List[str] = []
        options: List[str] = []
        for res in query_results:
            extracted_name = (res.get("extracted_name") or "").split()
            matches = res.get("matches", []) or []
            if not matches:
                continue

            high_results = [m for m in matches if m.get("score", 0) >= 0.8]
            mid_result = [m for m in matches if 0.8 > m.get("score", 0) >= 0.6]

            if len(high_results) == 1:
                confirmed_item_names.append(high_results[0].get("item_name"))
                continue

            if len(high_results) > 1:
                picked = None
                if extracted_name:
                    for hr in high_results:
                        if hr.get("item_name") == extracted_name:
                            picked = hr
                            break
                if not picked:
                    picked = high_results[0]

                confirmed_item_names.append(picked.get("item_name"))
                continue
            if len(mid_result) > 0:
                for mr in mid_result[:5]:
                    options.append(mr.get("item_name"))


        return {
            "confirmed_item_names": list(set(confirmed_item_names)),
            "options": list(set(options))
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        print("step_7：状态state信息，根据第6步高低分结果整理")
        confirmed = align_result.get("confirmed_item_names")
        options = align_result.get("options")

        if confirmed:
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    mid = msg.get("_id")
                    if mid:
                        ids_to_update.append(str(mid))

            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)

            state["item_names"] = confirmed
            state["answer"] = ""
            return state

        if options:
            state["item_names"] = []
            options_str = "、".join(options)
            state["answer"] = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
            return state

        if not confirmed and not options:
            state["answer"] = ""
            state["item_names"] = []

        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        print("step_8：写入历史会话：更新")

        if state.get("answer"):
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state["answer"],
                rewritten_query="",
                item_names=state.get("item_names", [])
            )

        save_chat_message(
            session_id=session_id,
            role="user",
            text=state["original_query"],
            rewritten_query=rewritten_query,
            item_names=state.get("item_names", []),
            message_id=message_id
        )


if __name__ == "__main__":
    init_state = {
        "original_query": "B530这个玩意咋捣鼓呀？",
        "session_id": "123"
    }

    node_item_name_confirm = NodeItemNameConfirm()
    result = node_item_name_confirm(init_state)
    logger.info(result)
