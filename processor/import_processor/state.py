"""

导入流程状态类型定义



定义完整的状态结构和辅助函数

"""

from typing import TypedDict, List

import copy


class ImportGraphState(TypedDict, total=False):


    """

    导入流程图状态



    包含整个导入流程中传递的所有数据

    """


    task_id: str


    is_md_read_enabled: bool

    is_pdf_read_enabled: bool


    import_file_path: str

    file_dir: str

    pdf_path: str

    md_path: str


    file_title: str

    item_name: str


    md_content: str

    chunks: List





GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "item_name": "",

}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
