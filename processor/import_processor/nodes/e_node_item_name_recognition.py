import json
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client import milvus_client

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):

        file_title, chunks = self._step_1_get_inputs(state)

        context = self._step_2_build_context(file_title, chunks)

        item_name = self._step_3_call_llm(file_title, context)
        print(item_name)

        self._step_4_update_chunks(state, chunks, item_name)
        path = f'{Path(state.get("md_path")).parent}/{state.get("file_title")}_new_new_chunks.json'
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        return state

    def _step_1_get_inputs(self, state) -> (str, List[Dict]):
        print("mode_item_name_recognition：步骤1：参数处理")
        file_title = state.get('file_title')
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get('chunks')
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)
        return file_title, chunks

    def _step_2_build_context(self, file_title, chunks: [Dict]) -> str:
        print("mode_item_name_recognition：步骤2：上下文拼接")
        k = self.config.item_name_chunk_k
        chunk_size = self.config.item_name_chunk_size

        parts: List[Dict] = []
        total_chunks = 0

        for index, chunk in enumerate(chunks[:k], start=1):
            chunk_title = chunk.get('title').strip()
            chunk_content = chunk.get('content').strip()

            piece = f"【切片{index}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            total_chunks += len(piece)

            if total_chunks >= chunk_size:
                break

        context = "\n\n".join(parts).strip()
        final_context = context[:chunk_size]
        return final_context

    def _step_3_call_llm(self, file_title, context):
        print("mode_item_name_recognition：步骤3：模型识别（总结）")

        if not context:
            return file_title

        llm_ai = ChatOpenAI(
            model=lm_config.llm_model,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            temperature=lm_config.llm_temperature,
            extra_body={"enable_thinking": True},
        )
        prompt = f"""
                    请从以下信息中识别出商品名称与型号：
                    文件名：{file_title}
                    
                    正文切片（用于辅助识别）：
                    {context}
                    
                    要求：
                    1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
                    2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
                    3. 如果无法识别商品名称,请返回空字符串。
                """

        message = [
            SystemMessage("你是一个专业的商品名称识别模型，请根据提供的信息，识别商品名称。名称不要多余20个字"),
            HumanMessage(prompt),
        ]

        response = llm_ai.invoke(message)

        item_name = getattr(response, "content", "").strip()
        item_name = item_name.replace(" ", "_").replace("\n", "").replace("\t", "").replace("\r", "")

        if not item_name:
            return file_title

        return item_name

    def _step_4_update_chunks(self, state, chunks, item_name):
        print("mode_item_name_recognition：步骤4：回填数据（item_name -> chunks）")
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name


        return state

    def _step_5_generate_vectors(self, item_name):
        print("mode_item_name_recognition：步骤5：主体名称向量化")

        embeddings = generate_embeddings([item_name])
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        return dense, sparse

    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        print("mode_item_name_recognition：步骤6：存入milvus向量库")

        client = get_milvus_client()
        collection_name = milvus_config.item_name_collection

        schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)

        schema.add_field(
            field_name="pk",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )
        schema.add_field(
            field_name="file_title",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=100
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 120},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",

                "normalize": True,

                "quantization": "none"
            })

        if not client.has_collection(collection_name):
            client.create_collection(collection_name=collection_name,schema=schema, index_params=index_params)

        safe_item_name = escape_milvus_string(item_name)
        filter_expr = f'item_name=="{safe_item_name}"'
        client.delete(collection_name=collection_name, filter=filter_expr)

        data = {
            "file_title": file_title,
            "item_name": item_name,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }

        client.insert(collection_name, [data])
        client.load_collection(collection_name)

        state["item_name"] = item_name
