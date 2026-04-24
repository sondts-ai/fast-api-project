# app/rag/crag.py
import re
import os
from typing import Any
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from domain.schema.GraphState import GraphState
from dotenv import load_dotenv
load_dotenv()
#LLM
model_name = os.getenv("OLLAMA_MODEL")
if model_name is None:
    raise ValueError ("No model name found !")
llm = ChatOllama(model=model_name, temperature=0.01)

def normalize_query(question: str) -> str:
    synonyms = {
        r"(?i)\bho guom | hồ hoàn kiếm | ho hoan kiem\b": " Hồ Hoàn Kiếm",
        r"(?i)\blăng bác | lang bac | lăng hồ chủ tịch | lang ho chu tich | lăng chủ tịch | lang chu tich\b": "Lăng Chủ tịch Hồ Chí Minh",
        r"(?i)\bvăn miếu | van mieu | quốc tử giám | quoc tu giam | van mieu quoc tu giam\b": "Văn Miếu Quốc Tử Giám",
        r"(?i)\bhoả lò | hoa lo | nhà tù hoả lò | nha tu hoa lo | di tích hoả lò | di tich hoa lo\b": "Di tích lịch sử Nhà tù Hỏa Lò",
        r"(?i)\bcột cờ | cot co\b": "Cột cờ Hà Nội",
        r"(?i)\bhoàng thành | hoang thanh | thành thăng long | thanh thang long | thăng long | thang long\b": "Hoàng thành Thăng Long",
        r"(?i)\b36 pho phuong | 36 phuong | 36 phường | pho phuong | phố phường\b": "36 phố phường",
        r"(?i)\bphố cổ | pho co | khu phố cổ | khu pho co | phố cổ hà nội | pho co ha noi\b": "Khu phố cổ Hà Nội"
    }
    # Duyệt qua từ điển và thay thế
    for pattern, replacement in synonyms.items():
        question = re.sub(pattern, replacement, question)
    # Cập nhật lại câu hỏi đã được "dịch" ra tên chuẩn
    print(f"\n[TIỀN XỬ LÝ] Câu hỏi sau chuẩn hoá: '{question}'")
    return question

# may_chem_van_phong — dọn output LLM
def may_chem_van_phong(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    text = text.replace('\\n', '\n')
    text = re.sub(r'[\u4e00-\u9fff]+', '', text).strip()
    pattern = r"^(Đúng rồi.*?|Bạn nói đúng.*?|Chính xác.*?|Dạ đúng.*?)(?:,|\.|\n|\s)+"
    while re.match(pattern, text, re.IGNORECASE):
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text[0].upper() + text[1:] if text else text

class CRAG:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def get_chain(self, vector_store: PGVector):

        # Node 1: retrieve
        async def retrieve_node(state: GraphState):
            question = state["question"]
            print(f"\n[RETRIEVE] Tìm kiếm: '{question}'")

            docs_with_scores = await vector_store \
                .asimilarity_search_with_relevance_scores(question, k=3)

            documents = [doc   for doc, _     in docs_with_scores]
            scores    = [score for _,   score in docs_with_scores]

            for doc, score in docs_with_scores:
                print(f"  score={score:.3f} | {doc.page_content[:60]}...")

            return {
                "documents":   documents,
                "scores":      scores,
                "question":    question,
                "retry_count": state.get("retry_count", 0),
            }

        # Node 2: grade
        def grade_node(state: GraphState):
            documents = state["documents"]
            scores    = state["scores"]

            relevant_docs = [
                doc for doc, score in zip(documents, scores)
                if score > self.threshold
            ]

            fallback = len(relevant_docs) == 0
            print(f"[GRADE] {len(relevant_docs)}/{len(documents)} docs vượt threshold {self.threshold}")

            return {
                "documents": relevant_docs,
                "fallback":  fallback,
            }

        # Node 3: rewrite
        async def rewrite_node(state: GraphState):
            question    = state["question"]
            documents   = state["documents"]  # docs đã có, không retrieve lại
            retry_count = state.get("retry_count", 0) + 1

            print(f"[REWRITE] Viết lại câu hỏi (retry={retry_count})")

            # Dùng docs cũ làm context để rewrite
            context = "\n\n".join(doc.page_content for doc in documents) \
                      if documents else ""

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                    "Bạn là người viết lại câu hỏi để tối ưu truy hồi dữ liệu.\n"
                    "Chỉ dùng thực thể và địa danh có trong <ngu_canh>.\n"
                    "Nếu không đủ thông tin, giữ nguyên câu hỏi gốc."
                ),
                ("human",
                    "<ngu_canh>\n{context}\n</ngu_canh>\n"
                    "Câu hỏi gốc: {question}\n"
                    "Viết lại ngắn gọn, rõ ý:"
                ),
            ])
            chain = prompt | llm | StrOutputParser()
            rewritten = await chain.ainvoke({
                "question": question,
                "context":  context,
            })
            rewritten = rewritten.strip() or question
            print(f"[REWRITE] '{question}' → '{rewritten}'")

            # Retrieve lại với câu hỏi mới
            docs_with_scores = await vector_store \
                .asimilarity_search_with_relevance_scores(rewritten, k=3)
            new_documents = [doc   for doc, _     in docs_with_scores]
            new_scores    = [score for _,   score in docs_with_scores]

            return {
                "question":    rewritten,
                "documents":   new_documents,
                "scores":      new_scores,
                "retry_count": retry_count,
            }

        # Node 4:
        async def generate_node(state: GraphState):
            question  = state["question"]
            documents = state["documents"]

            context = "\n\n".join(
                f"[{doc.metadata.get('source', 'Chung')}]\n{doc.page_content}"
                for doc in documents
            )

            prompt = PromptTemplate(
                template=(
                    "Bạn là NPC hướng dẫn viên du lịch ảo tại Hà Nội.\n"
                    "Chỉ trả lời dựa trên <ngu_canh> bên dưới.\n"
                    "TUYỆT ĐỐI không tự bịa thêm thông tin.\n\n"
                    "<ngu_canh>\n{context}\n</ngu_canh>\n\n"
                    "Câu hỏi: {question}\n"
                    "Trả lời:"
                ),
                input_variables=["context", "question"],
            )
            chain    = prompt | llm | StrOutputParser()
            response = await chain.ainvoke({
                "context":  context,
                "question": question,
            })
            print(f"[GENERATE] Done")
            return {"generation": may_chem_van_phong(response)}

        # Node 5: refuse
        def refuse_node(state: GraphState):
            print("[REFUSE] Không có thông tin phù hợp")
            return {
                "generation": "Tôi không có thông tin về vấn đề này."
            }

        # Conditional edge
        def decide(state: GraphState):
            if not state["fallback"]:
                return "generate"
            if state.get("retry_count", 0) >= 1:
                return "refuse"
            return "rewrite"

        # Build graph
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade",    grade_node)
        workflow.add_node("rewrite",  rewrite_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("refuse",   refuse_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges("grade", decide, {
            "generate": "generate",
            "rewrite":  "rewrite",
            "refuse":   "refuse",
        })
        workflow.add_edge("rewrite",  "grade")
        workflow.add_edge("generate", END)
        workflow.add_edge("refuse",   END)

        crag_app = workflow.compile()

        # Wrapper dùng trong RAGService
        async def run(raw_input: Any) -> str:
            normalized = normalize_query(raw_input)
            initial_state: GraphState = {
                "question":    normalized,
                "documents":   [],
                "scores":      [],
                "chat_history": [],
                "generation":  "",
                "fallback":    False,
                "retry_count": 0,
            }
            final_state = await crag_app.ainvoke(initial_state)
            return final_state["generation"]

        return RunnableLambda(run)