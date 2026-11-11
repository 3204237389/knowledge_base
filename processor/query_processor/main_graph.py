from dotenv import load_dotenv
from langgraph.graph import StateGraph

from processor.query_processor.nodes.a_node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.b_node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.c_node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.d_node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.nodes.e_node_rrf import NodeRrf
from processor.query_processor.nodes.f_node_rerank import NodeRerank
from processor.query_processor.nodes.g_node_answer_output import NodeAnswerOutput
from processor.query_processor.state import QueryGraphState

load_dotenv()

class KBQueryWorkflow:
    def __init__(self):
        self.workflow = StateGraph(QueryGraphState)

        self._init_nodes()

        self._register_nodes()

        self._setup_routes()

        self._compile_app = None

    def _init_nodes(self):
        print("实例化节点。。。")
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()
    def _register_nodes(self):
        print("注册节点。。。")
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer_output)

    def _setup_routes(self):
        print("添加边。。。")
        self.workflow.set_entry_point("node_item_name_confirm")

        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                "node_search_embedding":"node_search_embedding",
                "node_search_embedding_hyde":"node_search_embedding_hyde",
                "node_web_search_mcp":"node_web_search_mcp",
                "node_answer_output":"node_answer_output",
            }
        )

        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_embedding_hyde", "node_rrf")
        self.workflow.add_edge("node_web_search_mcp", "node_rrf")

        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")

    def _route_after_item_name_confirm(self, state:QueryGraphState)->str:
        print("路由条件。。。")
        if state.get("answer"):
            return "node_answer_output"

        return ["node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"]

    def compile(self):
        print("编译图。。。")
        if self._compile_app is None:
            self._compile_app = self.workflow.compile()

        return self._compile_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        print("调用图。。。")
        if not self._compile_app:
            self.compile()
        if stream:
            return self._compile_app.stream(initial_state)
        else:
            return self._compile_app.invoke(initial_state)


if __name__ == "__main__":
    workflow = KBQueryWorkflow()
    response = workflow.run({"original_query":"哥们儿，华为擎云B530计算机这个东东咋鼓捣上啊？","session_id":"123"},stream=True)
    for res in response:
        print(res)
