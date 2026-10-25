from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import anthropic
from config import CROSS_ENCODER
from pathlib import Path
from store import embed_chunk, collection

load_dotenv(Path(__file__).resolve().parent / ".env")
claude_client = anthropic.Anthropic()
reranker = CrossEncoder(CROSS_ENCODER)

TOP_K_RETRIEVED = 20   # dense 檢索撈幾個候選
TOP_K_RERANKED = 5     # rerank 後送幾條給 LLM
MAX_REF_ADD = 8        # 引用展開最多補幾條


#--- 條文索引 ---
#啟動時從 ChromaDB 讀出所有「條文」（表格列子 chunk 有 parent_id，不放進來）
#chunk_lookup：id → (條文內容, 來源檔名)
#article_refs：id → 它引用的條文 id（ingest 時解析好的）
#external_refs：id → 它引用的外部法規，例如「勞基法第12條」
#注意：重新跑 ingest.py 之後要重新啟動 rag.py，索引才會更新
def build_index():
    data = collection.get(include=["documents", "metadatas"])
    lookup, refs, external = {}, {}, {}
    for chunk_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if "parent_id" in meta:
            continue
        lookup[chunk_id] = (doc, meta["source"])
        refs[chunk_id] = meta["refs"].split("|") if meta.get("refs") else []
        external[chunk_id] = meta.get("ext_refs", "")
    return lookup, refs, external

chunk_lookup, article_refs, external_refs = build_index()
id_by_chunk = {chunk: chunk_id for chunk_id, chunk in chunk_lookup.items()}


#dense 檢索：回傳 top_k 個「條文」id
#子 chunk 命中時換成它所屬的條文；同一條文只算一次，所以多撈一些再去重
def dense_search(query: str, top_k: int) -> list[str]:
    query_embed = embed_chunk([query])[0]
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=min(top_k * 3, collection.count()),
        include=["metadatas"],
    )
    ids = []
    for chunk_id, meta in zip(results["ids"][0], results["metadatas"][0]):
        article_id = meta.get("parent_id", chunk_id)
        if article_id not in ids:
            ids.append(article_id)
        if len(ids) == top_k:
            break
    return ids

#輸入query及top_k數值，回傳top_k相近的條文及其source：[(片段內容, 來源檔名), ...]
def retrieve(query: str, top_k: int = TOP_K_RETRIEVED) -> list[tuple[str, str]]:
    return [chunk_lookup[i] for i in dense_search(query, top_k) if i in chunk_lookup]

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


#引用展開：rerank 前幾名的條文中，明確引用到的其他條文一定要一起送給 LLM
#只展開一層（被引用的條文再引用的不追）；依名次順序收集，最多 MAX_REF_ADD 條
def expand_references(chunks: list[tuple[str, str]], max_add: int = MAX_REF_ADD) -> list[tuple[str, str]]:
    present = {id_by_chunk.get(c) for c in chunks}
    added = []
    for c in chunks:
        for ref_id in article_refs.get(id_by_chunk.get(c), []):
            if ref_id in chunk_lookup and ref_id not in present and ref_id not in added:
                added.append(ref_id)
    return [chunk_lookup[i] for i in added[:max_add]]

def external_refs_of(chunk: tuple[str, str]) -> str:
    return external_refs.get(id_by_chunk.get(chunk), "")


#組出要送給 LLM 的內容：(rerank 前幾名, 被引用的條文)
def build_context(query: str, top_k_retrieved: int = TOP_K_RETRIEVED,
                  top_k_reranked: int = TOP_K_RERANKED) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    retrieved = retrieve(query, top_k_retrieved)
    top = rerank(query=query, retrieved_chunks=retrieved, top_k=top_k_reranked)
    return top, expand_references(top)



#取出回應中的文字：回應可能包含思考過程（thinking）等其他區塊，content[0] 不一定是文字
def response_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()


def format_chunk(chunk: tuple[str, str]) -> str:
    text, source = chunk
    note = external_refs_of(chunk)
    note = f"\n（此條文另依據外部法規：{note}；其內容未收錄）" if note else ""
    return f"【{source}】\n{text}{note}"

def generate(query: str, chunks: list[tuple[str, str]], referenced: list[tuple[str, str]] = ()) -> str:
    """chunks 是檢索到的條文，referenced 是它們引用的條文，格式都是 [(片段內容, 來源檔名), ...]"""
    system_prompt = (
        "你是一個人資知識助手，請根據用戶的問題和提供的相關片段生成準確回答。"
        "請只根據相關片段回答，不要編造資訊。如果相關片段不足以回答，就回答「不知道」。"
        "每個片段開頭的【】內是它所屬的規章名稱，回答時請註明依據的規章名稱，"
        "例如「依據 HR-04_差勤管理辦法」。"
        "片段分為兩部分：「檢索到的條文」是與問題最相關的條文；"
        "「被引用的條文」是檢索到的條文中明確引用的其他條文，用來補充說明前者。"
        "若條文依據外部法規（例如勞基法）而片段中沒有該法規內容，請說明是依據該法規，"
        "但不要自行補充或推論該法規的內容。"
    )

    if not chunks:
        return "不知道。目前的規章中找不到相關內容。"

    context = "【檢索到的條文】\n\n" + "\n\n".join(format_chunk(c) for c in chunks)
    if referenced:
        context += "\n\n【被引用的條文】\n\n" + "\n\n".join(format_chunk(c) for c in referenced)
    user_prompt = f"用戶問題：{query}\n\n相關片段：\n{context}"


    response = claude_client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response_text(response)

def answer(query: str, top_k_retrieved: int = TOP_K_RETRIEVED, top_k_reranked: int = TOP_K_RERANKED) -> str:
    top, referenced = build_context(query, top_k_retrieved, top_k_reranked)
    return generate(query, top, referenced)
