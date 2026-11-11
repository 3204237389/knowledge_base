import logging
from operator import truediv
from pathlib import Path

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        logging.info(f"{self.name}节点开始执行了...")

        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise StateFieldError(field_name='import_file_path', expected_type=str)
        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f"文件不存在:{import_file_path_obj.name}")
        if import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        elif import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        else:
            raise FileProcessingError(message=f"不支持的文件格式:{import_file_path_obj}")

        state["file_title"] = import_file_path_obj.stem
        state["file_dir"] = r"F:\output"
        return state
