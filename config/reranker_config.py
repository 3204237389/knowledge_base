from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    text_rerank_api_key: str
    text_rerank_model: str
    text_rerank_instruct: str

reranker_config = RerankerConfig(
    text_rerank_api_key=os.getenv("OPENAI_API_KEY"),
    text_rerank_model=os.getenv("TEXT_RERANK_MODEL"),
    text_rerank_instruct=os.getenv("TEXT_RERANK_INSTRUCT")
)
