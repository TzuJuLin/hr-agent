import os
import chromadb
from sentence_transformers import CrossEncoder 
from dotenv import load_dotenv
import anthropic
from config import CROSS_ENCODER
import re
from pathlib import Path
from store import embed_chunk,collection

load_dotenv(Path(__file__).resolve().parent / ".env")
claude_client = anthropic.Anthropic()
reranker = CrossEncoder(CROSS_ENCODER)




#輸入query及top_k數值，回傳計算後top_k相近的chunk及其source
def retrieve(query:str, top_k:int) -> list[tuple[str, str]]:
    query_embed = embed_chunk([query])[0]
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=top_k
    )
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(docs, sources))

#利用cross encoder重新排列打分數
#SentenceEncoder把每個chunk分別計算embedding，再比較向量之間的距離，且可存進ChromaDB中，速度較快
#CrossEncoder則把query和chunks都一起丟入模型中，較精準，但每次計算都是丟入（query+chunk)兩兩計算，因此速度很慢


#重新排列retrieve的chunks, 並得到top_k個結果，要給claude
def rerank(query: str, retrieved_chunks: list[tuple[str, str]], top_k: int) -> list[tuple[str, str]]:
    """retrieved_chunks 是 [(片段內容, 來源檔名), ...]"""
    if not retrieved_chunks:
        return []
    scores = reranker.predict([(query, text) for text, _ in retrieved_chunks])
    ranked = sorted(zip(retrieved_chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]



def generate(query: str, chunks: list[tuple[str, str]]) -> str:
    """chunks 是 [(片段內容, 來源檔名), ...]"""
    system_prompt = (
        "你是一個人資知識助手，請根據用戶的問題和提供的相關片段生成準確回答。"
        "請只根據相關片段回答，不要編造資訊。如果相關片段不足以回答，就回答「不知道」。"
        "每個片段開頭的【】內是它所屬的規章名稱，回答時請註明依據的規章名稱，"
        "例如「依據 HR-04_差勤管理辦法」。"
    )

    if not chunks:
        return "不知道。目前的規章中找不到相關內容。"

    context = "\n\n".join(f"【{source}】\n{text}" for text, source in chunks)
    user_prompt = f"用戶問題：{query}\n\n相關片段：\n{context}"


    response = claude_client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text

def answer(query: str, top_k_retrieved: int = 20, top_k_reranked: int = 3) -> str:
    retrieved_chunks = retrieve(query, top_k_retrieved)
    reranked_chunks = rerank(query=query, retrieved_chunks=retrieved_chunks, top_k=top_k_reranked)
    return generate(query, reranked_chunks)