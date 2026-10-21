from pathlib import Path

FOLDER_PATH='./docs'
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "hr_regulations"
CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
DB_NAME = Path(__file__).resolve().parent / "chroma_db"
