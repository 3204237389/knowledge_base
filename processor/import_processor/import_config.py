"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ImportConfig:
    """导入流程配置"""

    max_content_length: int = 2000
    img_content_length: int = 200
    min_content_length: int = 500
    overlap_sentences: int = 1
    item_name_chunk_k: int = 3
    item_name_chunk_size: int = 2500

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )


    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    requests_per_minute: int = 15

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self):
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + f"{self.minio_endpoint}"


_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
