import json
import os
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class CulturalEmbeddingService:
    def __init__(self, connection_string: str, collection_name: str, model_name: str = "keepitreal/vietnamese-sbert"):
        """
        Khởi tạo Service: Chuẩn bị máy xay thịt (Embedding Model) và Tủ lạnh (Database URL)
        """
        print("1. Khởi tạo Service và tải mô hình Embedding...")
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def load_documents_from_json(
        self,
        file_path: str,
        content_key: str = "content",
        metadata_key: str = "metadata",
    ) -> list[Document]:
        """
        Đọc file JSON và chuyển đổi thành danh sách các Document của Langchain
        """
        print(f"2. Đang đọc dữ liệu từ file {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file JSON: {e}")
            return []

        documents = []
        for item in data:
            page_content = item.get(content_key, "") or item.get("noi_dung", "")
            metadata_value = item.get(metadata_key)
            if isinstance(metadata_value, dict):
                metadata = metadata_value
            else:
                metadata = {"source": item.get("ten_dia_danh", "Unknown")}
            
            # Chỉ thêm vào nếu có nội dung thực sự
            if page_content.strip(): 
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)

        print(f"   -> Đã tạo xong {len(documents)} chunks.")
        return documents

    def ingest_to_db(self, documents: list[Document]):
        """
        Nhúng vector và lưu toàn bộ Document vào PostgreSQL
        """
        if not documents:
            print("⚠️ Không có tài liệu nào để nạp!")
            return

        print("3. Đang mã hóa Vector và lưu vào PostgreSQL...")
        try:
            # Dùng langchain_postgres để đồng bộ schema với runtime RAG trong project
            db = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
            )
            db.delete_collection()
            db.create_collection()
            db.add_documents(documents)
            print("✅ Hoàn tất! Dữ liệu đã được nạp thành công vào Database.")
        except Exception as e:
            print(f"❌ Lỗi khi lưu vào Database: {e}")

    def run_pipeline(self, file_path: str):
        """
        Hàm chạy toàn bộ quy trình từ A-Z
        """
        docs = self.load_documents_from_json(file_path)
        self.ingest_to_db(docs)



if __name__ == "__main__":
    # Cấu hình thông số
    DB_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:123456@db:5432/postgres",
    )
    COLLECTION = os.getenv("COLLECTION_NAME", "dia_danh_hanoi")
    
    # Mặc định phù hợp khi chạy trong container backend (WORKDIR=/app)
    JSON_FILE_PATH = os.getenv(
        "JSON_FILE_PATH",
        "resources/hanoi_knowledge_base_semantic.json",
    )

    # 1. Khởi tạo một "Công nhân" (Object) từ Class
    ingestion_service = CulturalEmbeddingService(
        connection_string=DB_URL, 
        collection_name=COLLECTION
    )

    # 2. Giao việc cho công nhân chạy full đường ống
    ingestion_service.run_pipeline(file_path=JSON_FILE_PATH)