import re
from typing import List, Dict, Tuple

from processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from utils.llm_utils import get_llm_client
from utils.mongo_history_utils import save_chat_message
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import set_task_result, add_done_task

MAX_CONTEXT_CHARS = 12000


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案输出
    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """

    name: str = "node_answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
        2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
        3 调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
        4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
        5 做最后一次push操作（主要是为了触发前端图片渲染)
             {
                "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
                "status": "completed",
                "image_urls": [
                    "http://local-server/images/panel_view.jpg",
                    "http://local-server/images/button_detail.jpg"
                ]
              }
        """
        answer_exists = self._step_1_check_answer(state)

        if not answer_exists:
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt

            self._step_3_generate_response(state, prompt)

        image_urls = self._extract_images_from_docs(state.get("reranked_docs") or [])

        if state.get("answer"):
            print("---写入MongoDB历史记录---")
            self._step_4_write_history(state, image_urls=image_urls)

        add_done_task(state['session_id'], self.name, state.get("is_stream"))

        print(f"---发送 final 事件---图片为：{image_urls}")
        if state.get("is_stream"):
            push_to_session(
                state['session_id'],
                SSEEvent.FINAL,
                {
                    "answer": state["answer"],
                    "status": "completed",
                    "image_urls": image_urls
                }
            )

        print("---node_answer_output 节点处理结束---")
        return state

    def _step_1_check_answer(self, state) -> bool:
        """
        阶段一：检查 state 中是否已有 answer。
        - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
        - 若不存在：返回 False
        """
        answer = state.get("answer")
        is_stream = state.get("is_stream")
        if answer:
            if is_stream:
                print("---Step 1: 发现已有答案，执行流式推送---")
                push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
            else:
                set_task_result(state["session_id"], "answer", answer)
            return True
        else:
            return False

    def _step_2_construct_prompt(self, state: QueryGraphState) -> str:

        """
        阶段二：构建 Prompt
        根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组装 LLM 提示词
        """
        char_budget = MAX_CONTEXT_CHARS

        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        item_names_str = ", ".join(item_names) if item_names else "无指定商品"

        prompt = ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=item_names_str,
            question=question,
        )
        print(f"组装后的提示词为：{prompt}")
        return prompt

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, start=1):
            content = doc.get("content")
            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field)).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _step_3_generate_response(self, state: QueryGraphState, prompt: str) -> QueryGraphState:
        """
        阶段三：生成回答
        调用llm生成答案，支持流式输出
        """
        print("---Step 3: 开始生成回答 (LLM Generation)---")

        llm = get_llm_client()

        session_id = state.get("session_id")
        is_stream = state.get("is_stream")

        if is_stream:
            print(f"模式: 流式输出 (Streaming), Session: {session_id}")
            final_text = ""
            try:
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    if delta:
                        final_text += delta
                        push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})

                print(f"流式输出完成，总长度: {len(final_text)}")

            except Exception as e:
                print(f"流式生成出错: {e}", exc_info=True)
                push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})

            state["answer"] = final_text
        else:
            print(f"模式: 非流式输出 (Blocking), Session: {session_id}")
            try:
                response = llm.invoke(prompt)
                content = response.content
                state["answer"] = content
                set_task_result(session_id, "answer", content)
                print(f"生成回答完成，长度: {len(content)}")
            except Exception as e:
                print(f"生成回答出错: {e}", exc_info=True)
                state["answer"] = "抱歉，生成回答时出现错误。"

        return state

    def _extract_images_from_docs(self, docs):
        """
        辅助方法：从文档列表中提取图片URL

        核心逻辑：
        1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
        2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
           - 验证后缀名是否为图片格式 (.jpg, .png 等)。
        3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
           - 匹配 Markdown 图片语法: ![alt text](image_url)。
        4. 对提取到的 URL 进行去重处理，返回唯一图片列表。

        :param docs: 文档列表，每个文档为字典格式
        :return: 图片 URL 字符串列表
        """
        images = []
        seen = set()
        if not docs:
            return []
        md_img_pattern = re.compile(r'!\[.*?\]\((.*?\.(?:png|jpg|jpeg|gif|webp|bmp|svg))\)')
        print(f"开始提取图片，待处理文档数: {len(docs)}")

        for i, doc in enumerate(docs):
            url = (doc.get("url") or "").strip()
            if url:
                if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                    if url not in seen:
                        print(f"文档[{i}] 发现图片 URL (字段): {url}")
                        seen.add(url)
                        images.append(url)

            text = (doc.get("content") or "").strip()
            if text:
                matches = md_img_pattern.findall(text)
                for img_url in matches:
                    img_url = img_url.strip()
                    if img_url and img_url not in seen:
                        print(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                        seen.add(img_url)
                        images.append(img_url)

        print(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")
        return images

    def _step_4_write_history(seld, state: QueryGraphState, image_urls=None) -> QueryGraphState:
        """
        阶段四：把本轮答案写入 MongoDB history。
        利用 utils/mongo_history_utils.py 中的 save_chat_messages 方法。
        """
        session_id = state.get("session_id", "default")
        answer = (state.get("answer") or "").strip()
        item_names = state.get("item_names") or []

        try:
            if answer:
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    item_names=item_names,
                    image_urls=image_urls,
                    message_id=None
                )
        except Exception as e:
            print(f"写入Mongo历史记录失败: {e}")

        return state


if __name__ == "__main__":
    print("开始测试: 答案生成节点")

    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2",
            "score": 0.95,
            "url": None,
            "content": """
HAK 180 烫金机的操作面板位于机器正前方。
开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
具体的操作面板布局请参考下图：
![操作面板布局图](http://192.168.100.100:9000/knowledge-base/upload-images/hak180产品安全手册/048c005b198be5c9fff80ad6a6ba02496f38fa109ec20dbaabde3110f3eb1574.jpg)

如果是进行局部烫金，请调节侧面的旋钮。
![侧面旋钮细节](http://192.168.100.100:9000/knowledge-base/upload-images/hak180产品安全手册/f77da4df52517fc50b9efb528540e1351dd1a08dce6f801cf08366540f2c59ce.jpg)
"""
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://192.168.100.100:9000/knowledge-base/upload-images/%E5%8D%8E%E4%B8%BA%E6%93%8E%E4%BA%91G740%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97-(KLVG-16Z,Windows11_02,zh-cn)/c28a751c315a89fb5f3b52736a7996b56971c9a260a0e2b850eb5ef18beabf3c.jpg",
            "content": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "url": None,
            "content": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK180烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None
    }

    node_answer_output = NodeAnswerOutput()
    result = node_answer_output(mock_state)

    print(result)
