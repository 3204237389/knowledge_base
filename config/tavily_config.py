from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TavilyConfig:
    api_key: str


tavily_config = TavilyConfig(
    api_key=os.getenv("TAVILY_API_KEY")
)