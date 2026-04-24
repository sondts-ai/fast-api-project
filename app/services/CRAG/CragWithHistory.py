import re
import os
from typing import Any
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
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
        r"(?i)\blăng bác | lang bac | lăng hồ chủ tịch | lang ho chu tich | lăng chủ tịch | lang chu tich\b": " Lăng Chủ tịch Hồ Chí Minh",
        r"(?i)\bvăn miếu | van mieu | quốc tử giám | quoc tu giam | van mieu quoc tu giam\b": " Văn Miếu Quốc Tử Giám",
        r"(?i)\bhoả lò | hoa lo | nhà tù hoả lò | nha tu hoa lo | di tích hoả lò | di tich hoa lo\b": " Di tích lịch sử Nhà tù Hỏa Lò",
        r"(?i)\bcột cờ | cot co\b": "Cột cờ Hà Nội",
        r"(?i)\bhoàng thành | hoang thanh | thành thăng long | thanh thang long | thăng long | thang long\b": " Hoàng thành Thăng Long",
        r"(?i)\b36 pho phuong | 36 phuong | 36 phường | pho phuong | phố phường\b": " 36 phố phường",
        r"(?i)\bphố cổ | pho co | khu phố cổ | khu pho co | phố cổ hà nội | pho co ha noi\b": " Khu phố cổ Hà Nội"
    }
    for pattern, replacement in synonyms.items():
        question = re.sub(pattern, replacement, question)
    print(f"\n[TIỀN XỬ LÝ] Câu hỏi sau chuẩn hoá: '{question}'")
    return question


def may_chem_van_phong(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    text = text.replace('\\n', '\n')
    text = re.sub(r'[\u4e00-\u9fff]+', '', text).strip()
    pattern = r"^(Đúng rồi.*?|Bạn nói đúng.*?|Chính xác.*?|Dạ đúng.*?)(?:,|\.|\n|\s)+"
    while re.match(pattern, text, re.IGNORECASE):
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text[0].upper() + text[1:] if text else text


class CRAGWithHistory:
    def __init__(self):
        pass

    def get_chain(self, vector_store: PGVector):
        async def retrieve_node(state: GraphState):
            question = state["question"]
            print(f"\n[RETRIEVE] Tìm kiếm: '{question}'")

            documents = await vector_store.amax_marginal_relevance_search(
                question, k=3, fetch_k=10
            )

            return {
                "documents": documents,
                "scores": [1.0] * len(documents),
                "question": question,
                "chat_history": state.get("chat_history", []),
                "retry_count": state.get("retry_count", 0),
            }

        async def grade_node(state: GraphState):
            documents = state["documents"]
            question = state["question"]
            prompt = PromptTemplate(
                template=(
                    "Bạn là một trợ lý kiểm tra dữ liệu. Hãy xem tài liệu có chứa thông tin để trả lời câu hỏi hay không.\n"
                    "Chỉ trả lời \"yes\" hoặc \"no\". Không giải thích gì thêm.\n\n"
                    "Ví dụ 1:\n"
                    "TÀI LIỆU: Hồ Gươm nằm ở trung tâm thủ đô Hà Nội.\n"
                    "CÂU HỎI: Hồ Gươm ở đâu?\n"
                    "TRẢ LỜI: yes\n\n"
                    "Ví dụ 2:\n"
                    "TÀI LIỆU: Phở là món ăn truyền thống của Việt Nam.\n"
                    "CÂU HỎI: Ai là người xây dựng Văn Miếu?\n"
                    "TRẢ LỜI: no\n\n"
                    "Bây giờ đến lượt bạn:\n"
                    "TÀI LIỆU: {document}\n"
                    "CÂU HỎI: {question}\n"
                    "TRẢ LỜI:"
                ),
                input_variables=["document", "question"],
            )
            grader_chain = prompt | llm | StrOutputParser()
            relevant_docs = []
            for doc in documents:
                score_response = await grader_chain.ainvoke({
                    "question": question,
                    "document": doc.page_content,
                })
                if "yes" in score_response.strip().lower():
                    relevant_docs.append(doc)

            fallback = len(relevant_docs) == 0
            print(f"[GRADE] LLM lọc giữ lại {len(relevant_docs)}/{len(documents)} docs")

            return {
                "documents": relevant_docs,
                "fallback": fallback,
                "chat_history": state.get("chat_history", []),
            }

        async def rewrite_node(state: GraphState):
            question = state["question"]
            retry_count = state.get("retry_count", 0) + 1

            print(f"[REWRITE] Viết lại câu hỏi (retry={retry_count})")

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Hãy trích xuất từ khóa quan trọng nhất từ câu hỏi dưới đây để tìm kiếm. Chỉ in ra từ khóa, không in gì thêm."),
                ("human", "Câu hỏi gốc: {question}\nTừ khóa:"),
            ])
            chain = prompt | llm | StrOutputParser()
            rewritten = await chain.ainvoke({"question": question})
            rewritten = rewritten.strip() or question
            print(f"[REWRITE] '{question}' → Keyword: '{rewritten}'")

            new_documents = await vector_store.amax_marginal_relevance_search(
                rewritten, k=3, fetch_k=10
            )

            return {
                "question": rewritten,
                "documents": new_documents,
                "scores": [1.0] * len(new_documents),
                "chat_history": state.get("chat_history", []),
                "retry_count": retry_count,
            }

        async def generate_node(state: GraphState):
            question = state["question"]
            documents = state["documents"]
            chat_history = state.get("chat_history", [])

            context = "\n\n".join(
                f"[{doc.metadata.get('source', 'Chung')}]\n{doc.page_content}"
                for doc in documents
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Bạn là một NPC hướng dẫn viên du lịch ảo nhiệt tình, am hiểu sâu sắc về văn hóa, lịch sử và địa danh Hà Nội.\n"
                 "Nhiệm vụ của bạn là giải đáp thắc mắc cho du khách một cách tự nhiên dựa trên <ngu_canh> dưới đây.\n"
                 "Bạn được phép dùng <chat_history> để hiểu ngữ cảnh hội thoại.\n\n"
                 "<ngu_canh>\n{context}\n</ngu_canh>\n\n"
                 "Hướng dẫn trả lời:\n"
                 "- Hãy trả lời ngắn gọn, lịch sự và chính xác những gì có trong <ngu_canh>.\n"
                 "- TUYỆT ĐỐI KHÔNG tự ý suy diễn hay bịa đặt thông tin không có trong <ngu_canh>.\n"
                 "- Nếu <ngu_canh> không có thông tin, bạn BẮT BUỘC trả lời: 'Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác'."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])

            chain = prompt | llm | StrOutputParser()
            response = await chain.ainvoke({
                "context": context,
                "question": question,
                "chat_history": chat_history,
            })
            print("[GENERATE] Done")
            return {"generation": may_chem_van_phong(response)}

        def refuse_node(state: GraphState):
            print("[REFUSE] Không có thông tin phù hợp")
            return {"generation": "Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác."}

        def decide(state: GraphState):
            if not state["fallback"]:
                return "generate"
            if state.get("retry_count", 0) >= 1:
                return "refuse"
            return "rewrite"

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade", grade_node)
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("refuse", refuse_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges("grade", decide, {
            "generate": "generate",
            "rewrite": "rewrite",
            "refuse": "refuse",
        })
        workflow.add_edge("rewrite", "grade")
        workflow.add_edge("generate", END)
        workflow.add_edge("refuse", END)

        crag_app = workflow.compile()

        async def run(payload: Any) -> str:
            if isinstance(payload, dict):
                raw_input = str(payload.get("input", ""))
                chat_history = payload.get("chat_history", [])
            else:
                raw_input = str(payload)
                chat_history = []

            normalized = normalize_query(raw_input)
            initial_state: GraphState = {
                "question": normalized,
                "documents": [],
                "scores": [],
                "chat_history": chat_history,
                "generation": "",
                "fallback": False,
                "retry_count": 0,
            }
            final_state = await crag_app.ainvoke(initial_state)
            return final_state["generation"]

        return RunnableLambda(run)
