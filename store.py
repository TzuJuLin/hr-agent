from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, COLLECTION_NAME, DB_NAME
import chromadb

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
chromadb_client = chromadb.PersistentClient(DB_NAME)
collection = chromadb_client.get_or_create_collection(name=COLLECTION_NAME)

#把每個chunk轉換成向量
def embed_chunk(chunk:list[str]) -> list[list[float]]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()
