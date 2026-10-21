import os
import chromadb
from sentence_transformers import SentenceTransformer
from config import FOLDER_PATH
import re
from store import embed_chunk, embedding_model, collection

#把文章切成chunk
#回傳值為 a list of dicts 範例 [{id: "HR-01_工作規則.md-1", text:"### 第1條（目的）[...]。", source:"HR-01_工作規則.md"}, ...]
# def split_into_chunks(doc_file: str) -> list[dict]:
#     with open(doc_file, "r", encoding="utf-8") as file:
#         content = file.read()
#     #保留來源文件，for reference
#     source = os.path.basename(doc_file)  
#     chunks = [c.strip() for c in re.split(r"\n(?=### )", content) if c.strip()]
#     return [
#         {"id": f"{source}-{i}", "text": chunk, "source": source}
#         for i, chunk in enumerate(chunks)
#     ]
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

def split_into_chunks(doc_file: str) -> list[dict]:
    with open(doc_file, "r", encoding="utf-8") as file:
        content = file.read()
    source = os.path.basename(doc_file)

    # 1. 移除開頭的 front matter，順便取出文件標題
    title = source
    m = FRONT_MATTER.match(content)
    if m:
        content = content[m.end():]
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.MULTILINE)
        if t:
            title = t.group(1).strip()

    # 2. 逐行讀取：記錄目前的章名，每個 ### 開始一條新條文
    articles = []
    chapter = ""
    current = None
    for line in content.splitlines():
        if line.startswith("## "):        # 章標題：記下來，但不放進任何條文
            chapter = line[3:].strip()
            current = None
        elif line.startswith("### "):     # 條文標題：開始新的一條
            current = {"chapter": chapter, "lines": [line]}
            articles.append(current)
        elif line.startswith("# "):       # 文件大標題：丟掉
            current = None
        elif current is not None:         # 條文內容
            current["lines"].append(line)

    # 3. 組成 chunk：text 是條文原文，embed_text 額外加上文件與章節脈絡
    chunks = []
    for i, a in enumerate(articles):
        text = "\n".join(a["lines"]).strip()
        header = f"【{title}｜{a['chapter']}】" if a["chapter"] else f"【{title}】"
        chunks.append({
            "id": f"{source}-{i}",
            "text": text,
            "embed_text": f"{header}\n{text}",
            "source": source,
        })
    return chunks

#把得到的向量存入向量資料庫中
def save_embeddings(chunks: list[dict], embeddings:list[list[float]]) -> None:
    collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c["source"]} for c in chunks],
        )
    print(f"已寫入 {len(chunks)} 段，資料庫共 {collection.count()} 段")



def main():
    folder_path = FOLDER_PATH
    list_of_chunk = []
    #go through 整個 FOLDER_PATH (文件所在資料夾）
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".md"):
                continue
            file_path = os.path.join(root, file)
            list_of_chunk += split_into_chunks(file_path)

    #取出list of chunk當中的文字部分
    texts = [c["text"] for c in list_of_chunk]

    #得到每個chunk的embeddings
    #embeddings = embedding_model.encode(texts, normalize_embeddings=True).tolist()
    embeddings = embed_chunk(texts)

    #把文件跟提取的向量放到向量數據庫中
    save_embeddings(list_of_chunk, embeddings)

if __name__ == "__main__":
    main()