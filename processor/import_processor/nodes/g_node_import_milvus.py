import json
from typing import Dict, Any, List

from pymilvus import MilvusClient, DataType

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):

        chunks_json_data,vector_dimension = self._step_1_check_inputs(state)

        milvus_client = self._step_2_prepare_collection(vector_dimension)

        self._step_3_clean_old_data(milvus_client, chunks_json_data)

        updated_chunks = self._step_4_insert_data(milvus_client, chunks_json_data)


        return state

    def _step_1_check_inputs(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        chunks_json_data = state.get("chunks")
        if not chunks_json_data:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        dimenstion = len(chunks_json_data[0]["dense_vector"])
        print(f"向量维度：{dimenstion}")

        return chunks_json_data, dimenstion

    def _step_2_prepare_collection(self, vector_dimension: int):
        print("node_import_milvus：步骤2：milvus客户端和集合准备")
        milvus_client = get_milvus_client()

        collection_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self._create_chunks_collection(collection_name, milvus_client, vector_dimension)
        return milvus_client

    def _step_3_clean_old_data(self, client, chunks_json_data):
        print("node_import_milvus：步骤3：清理可能的冗余数据（幂等性）")
        file_title = chunks_json_data[0].get("file_title")

        self.clear_chunks_by_file_title(client, file_title)


    def _step_4_insert_data(self, milvus_client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("node_import_milvus：步骤4：数据入库，返回数据库主键")

        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()
            if "part" not in item_copy:
                item_copy["part"] = 0
            data_to_insert.append(item_copy)
        insert_result = milvus_client.insert(collection_name=milvus_config.chunks_collection,data=data_to_insert)
        insert_count = insert_result.get("insert_count")
        print("插入{insert_count}条数据")

        insert_ids = insert_result.get("ids")
        for index,item in enumerate(chunks_json_data):
            item["chunk_id"] = str(insert_ids[index])
        return chunks_json_data

    def _create_chunks_collection(self, collections_name, milvus_client, vector_dimension):
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params
        )

    def clear_chunks_by_file_title(self, client, file_title):
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")
