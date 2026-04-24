import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict, messages_to_dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import cast
from repositories.RAGRepository import RAGRepository
from domain.schema.Ai import ChatRequest
from fastapi import Depends, HTTPException
from domain.schema.RagChatHistory import ChatHistoryUpdate
from langchain_postgres.vectorstores import PGVector
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from services.CRAG.CragWithHistory import CRAGWithHistory
from dotenv import load_dotenv
load_dotenv()
# RAG define
model_name = os.getenv("OLLAMA_MODEL")
if model_name is None:
    raise ValueError ("No model name found !")
llm = ChatOllama(model=model_name,temperature=0.01)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
if EMBEDDING_MODEL_NAME is None :
    raise ValueError("No embedding model found !")
async_connection = os.getenv("A_DATABASE_URL")
if async_connection is None:
    raise ValueError("A_DATABASE_URL is required for async PGVector operations")
prompt_template = ChatPromptTemplate.from_messages(
    [
    (
        "system", # ràng buộc chặt chẽ hơn
        "Bạn là một hướng dẫn viên du lịch ảo tại Hà Nội. BẠN BỊ CẤM SỬ DỤNG KIẾN THỨC BÊN NGOÀI.\n"
        "Ưu tiên sử dụng dữ liệu trong <ngu_canh> để trả lời câu hỏi kiến thức.\n"
        "Bạn được phép dùng <chat_history> để hiểu ngữ cảnh hội thoại và trả lời các câu hỏi về lịch sử trò chuyện\n"
        "(ví dụ: 'tôi vừa hỏi gì', 'bạn vừa trả lời gì').\n\n"
        "<ngu_canh>\n{context}\n</ngu_canh>\n\n"
        "QUY TẮC BẮT BUỘC: Nếu không có thông tin trong <ngu_canh> và cũng không có trong <chat_history>, "
        "TUYỆT ĐỐI KHÔNG TỰ BỊA ĐẶT mà hãy trả lời nguyên văn: "
        "'Tôi không biết, tôi không có thông tin về nó và không trả lời thêm thông tin nào khác'.",
    ),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{input}"),
    ]
)
embedding = HuggingFaceEndpointEmbeddings(huggingfacehub_api_token=HF_TOKEN, model=EMBEDDING_MODEL_NAME)
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

class RagService:
    _retriever_cache = {}
    _vector_store_cache = {}
    _crag_chain_cache = {}
    def __init__(self, repo : RAGRepository = Depends()):
        self.repo = repo

    async def get_vector_store(self, collection_name: str):
        if collection_name in self._vector_store_cache:
            return self._vector_store_cache[collection_name]
        vector = PGVector(
            embedding,
            connection=async_connection,
            collection_name=collection_name,
            async_mode=True,
            create_extension=False,
        )
        self._vector_store_cache[collection_name] = vector
        return vector

    async def get_retriever (self, collection_name : str) :
        if collection_name in self._retriever_cache:
            return self._retriever_cache[collection_name]
        vector = await self.get_vector_store(collection_name)
        retriever = vector.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        self._retriever_cache[collection_name] = retriever
        return retriever


    async def ask(self, user_id: int, chat_id: int, collection_name: str, request: ChatRequest):
        if not await self.repo.exist_by_collecton_name(collection_name):
            raise HTTPException(status_code=404, detail="Collection or document name not found !")

        orm_obj = await self.get_chat_history_by_id(chat_id)
        if not orm_obj or cast(int, orm_obj.user_id) != user_id:
            raise HTTPException(status_code=404, detail="Chat history not found !")

        retriever = await self.get_retriever(collection_name)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        chat_history = messages_from_dict(cast(list, orm_obj.messages))
        response = await chain.ainvoke({
            "input": request.prompt,
            "chat_history": chat_history,
        })
        answer = str(response["answer"])

        chat_history.append(HumanMessage(content=request.prompt))
        chat_history.append(AIMessage(content=answer))
        updated_messages = messages_to_dict(chat_history)
        history_update = ChatHistoryUpdate(id=cast(int, orm_obj.id), messages=updated_messages)
        await self.repo.update_chat_history(history_update)
        print(answer)
        return answer
    # None history
    async def ask_with_base_crag(self, collection_name : str, request: ChatRequest):
        vector_store = await self.get_vector_store(collection_name)
        if collection_name not in self._crag_chain_cache:
            self._crag_chain_cache[collection_name] = CRAGWithHistory().get_chain(vector_store)
        chain = self._crag_chain_cache[collection_name]
        answer = await chain.ainvoke(request.prompt)
        print(answer)
        return answer
    # With history logic
    async def ask_with_crag(self, user_id: int, chat_id: int, collection_name: str, request: ChatRequest):
        if not await self.repo.exist_by_collecton_name(collection_name):
            raise HTTPException(status_code=404, detail="Collection or document name not found !")

        orm_obj = await self.get_chat_history_by_id(chat_id)
        if not orm_obj or cast(int, orm_obj.user_id) != user_id:
            raise HTTPException(status_code=404, detail="Chat history not found !")

        vector_store = await self.get_vector_store(collection_name)
        if collection_name not in self._crag_chain_cache:
            self._crag_chain_cache[collection_name] = CRAGWithHistory().get_chain(vector_store)
        chain = self._crag_chain_cache[collection_name]

        chat_history = messages_from_dict(cast(list, orm_obj.messages))
        answer = await chain.ainvoke({
            "input": request.prompt,
            "chat_history": chat_history
        })

        chat_history.append(HumanMessage(content=request.prompt))
        chat_history.append(AIMessage(content=str(answer)))
        updated_messages = messages_to_dict(chat_history)
        history_update = ChatHistoryUpdate(id=cast(int, orm_obj.id), messages=updated_messages)
        await self.repo.update_chat_history(history_update)
        print(str(answer))
        return str(answer)

    async def get_create_chat(self, user_id : int):
        return await self.repo.create_chat_history(user_id)
    async def get_chat_histories_by_user_id (self,user_id : int):
        return await self.repo.find_by_user_id(user_id)
    async def get_update_chat_history(self, history: ChatHistoryUpdate):
        return await self.repo.update_chat_history(history)
    async def is_chat_history_with_user_id_exist(self, user_id : int, chat_id : int):
        return await self.repo.exist_by_chat_history_and_user_id(chat_id, user_id)
    async def get_chat_history_by_id(self, id: int):
        return await self.repo.find_by_id(id)
    async def get_delete (self, id : int) :
        return await self.repo.delete_by_id(id)

