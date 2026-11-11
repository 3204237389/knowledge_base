import json
from pathlib import Path
from typing import Dict


from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        chunks = self._step_1_validate_paths(state)

        output_data = self._step_2_generate_embeddings(chunks)
        path = f'{Path(state.get("md_path")).parent}/{state.get("file_title")}_new_new_new_chunks.json'
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=2,
            )
        state["chunks"] = output_data

        return state

    def _step_1_validate_paths(self, state):
        print("node_bge_embedding:步骤1：参数校验")

        chunks = state.get("chunks")
        if not chunks:
            raise Exception("参数错误：chunks")

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
        return chunks

    def _step_2_generate_embeddings(self, chunks: Dict[str, str]):
        """
        将item_name,content转化为向量数据（稠密和稀疏）
        """
        print("node_bge_embedding:步骤2：数据向量化")
        output_data = []

        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            five_ready_xlh_texts = []
            five_texts = chunks[i:i + batch_size]
            for doc in five_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                five_ready_xlh_texts.append(f"{item_name}\n{content}" if item_name else content)

            embeddings = generate_embeddings(five_ready_xlh_texts)

            for j,doc in enumerate(five_texts):
                item = doc.copy()
                dense = embeddings["dense"][j]
                item["dense_vector"] = dense
                sparse = embeddings["sparse"][j]
                item["sparse_vector"] = sparse
                output_data.append(item)




        return output_data
