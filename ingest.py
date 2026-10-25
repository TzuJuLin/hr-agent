import csv
import os
from config import FOLDER_PATH
import re
from pathlib import Path
from store import embed_chunk, collection
from refs import article_number, parse_references

REFS_REPORT = Path(__file__).resolve().parent / "refs_report.csv"

#把文章切成chunk
#回傳值為 a list of dicts 範例 [{id: "HR-01_工作規則.md-1", text:"### 第1條（目的）[...]。", title:"工作規則", source:"HR-01_工作規則.md"}, ...]
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

    # 2. 逐行讀取：每個 ### 開始一條新條文
    articles = []
    current = None
    for line in content.splitlines():
        if line.startswith("### "):       # 條文標題：開始新的一條
            current = [line]
            articles.append(current)
        elif line.startswith(("## ", "# ")):   # 章標題、文件大標題：不放進任何條文
            current = None
        elif current is not None:         # 條文內容
            current.append(line)

    # 3. 組成 chunk：text 是條文原文
    return [
        {"id": f"{source}-{i}", "text": "\n".join(lines).strip(), "title": title, "source": source}
        for i, lines in enumerate(articles)
    ]


#--- 條文引用 ---
#解析每條條文引用了哪些條文，存進 metadata：
#  refs：被引用的內部條文 chunk id，以 | 分隔（rag.py 用來把被引用的條文一起送給 LLM）
#  ext_refs：外部法規，例如「勞基法第12條」（資料庫沒有收錄，只標註）
#同時輸出 refs_report.csv，方便人工檢查解析結果
def doc_code(source: str) -> str:
    return source.split("_")[0]          # "HR-01_工作規則.md" → "HR-01"

def heading(text: str) -> str:
    return text.strip().splitlines()[0].lstrip("#").strip()

def attach_references(articles: list[dict]) -> None:
    titles = {a["title"]: doc_code(a["source"]) for a in articles}
    index = {(doc_code(a["source"]), article_number(a["text"])): a["id"] for a in articles}

    report = []
    for a in articles:
        internal, external = parse_references(a["text"], doc_code(a["source"]), titles)
        found = [index[r] for r in internal if r in index]
        missing = [f"{d} 第{n}條" for d, n in internal if (d, n) not in index]
        a["refs"] = "|".join(found)
        a["ext_refs"] = "、".join(external)
        if internal or external:
            report.append({
                "條文": f"{doc_code(a['source'])} {heading(a['text'])}",
                "內部引用": "、".join(f"{d} 第{n}條" for d, n in internal if (d, n) in index),
                "外部法規": a["ext_refs"],
                "找不到的引用": "、".join(missing),
            })

    with open(REFS_REPORT, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig：Excel 開啟不會亂碼
        writer = csv.DictWriter(f, fieldnames=["條文", "內部引用", "外部法規", "找不到的引用"])
        writer.writeheader()
        writer.writerows(report)
    n_missing = sum(bool(r["找不到的引用"]) for r in report)
    print(f"{len(report)} 條條文有引用，{n_missing} 條有找不到的引用；清單：{REFS_REPORT.name}")


#--- 表格列子 chunk ---
#像 HR-02 第3條（薪資項目）這種表格條文，一條裡面有很多不相關的項目，整條做 embedding 會被稀釋。
#所以把表格的每一列另外做成一個子 chunk 只拿來做 embedding；檢索到子 chunk 時，送出的仍是整條條文。
#例：「第3條（薪資項目） 項目：專業證照津貼；說明：持有公司公告之證照且工作上使用者；金額：每月 1,000 至 3,000 元」
TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEPARATOR = re.compile(r"^[\s|:\-]+$")

def table_row_chunks(parent: dict) -> list[dict]:
    lines = parent["text"].splitlines()
    rows = [l for l in lines if TABLE_ROW.match(l) and not TABLE_SEPARATOR.match(l)]
    if len(rows) < 2:                               # 沒有表格（至少要有表頭 + 一列）
        return []
    cells = lambda row: [c.strip() for c in row.strip().strip("|").split("|")]
    columns = cells(rows[0])
    children = []
    for j, row in enumerate(rows[1:]):
        desc = "；".join(f"{col}：{val}" for col, val in zip(columns, cells(row)) if val)
        children.append({
            "id": f"{parent['id']}-row{j}",
            "text": f"{heading(parent['text'])} {desc}",
            "parent_id": parent["id"],
            "source": parent["source"],
        })
    return children


#把得到的向量存入向量資料庫中
#條文的 metadata：source，有引用時加上 refs / ext_refs
#子 chunk 的 metadata：source、parent_id（rag.py 用來換回整條條文）
def metadata_of(c: dict) -> dict:
    if "parent_id" in c:
        return {"source": c["source"], "parent_id": c["parent_id"]}
    meta = {"source": c["source"]}
    if c.get("refs"):
        meta["refs"] = c["refs"]
    if c.get("ext_refs"):
        meta["ext_refs"] = c["ext_refs"]
    return meta

def save_embeddings(chunks: list[dict], embeddings:list[list[float]]) -> None:
    # 刪掉這次沒出現的舊 chunk（例如條文被刪除或檔案改名），避免資料庫殘留過期內容
    new_ids = {c["id"] for c in chunks}
    stale_ids = [i for i in collection.get(include=[])["ids"] if i not in new_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[metadata_of(c) for c in chunks],
        )
    n_children = sum("parent_id" in c for c in chunks)
    print(f"已寫入 {len(chunks) - n_children} 條條文 + {n_children} 個表格列，"
          f"刪除 {len(stale_ids)} 段舊資料，資料庫共 {collection.count()} 段")



def main():
    folder_path = FOLDER_PATH
    articles = []
    #go through 整個 FOLDER_PATH (文件所在資料夾）
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".md"):
                continue
            file_path = os.path.join(root, file)
            articles += split_into_chunks(file_path)

    #解析條文引用，並輸出 refs_report.csv
    attach_references(articles)

    #表格條文另外產生每一列的子 chunk
    list_of_chunk = articles + [row for a in articles for row in table_row_chunks(a)]

    #embedding 只用條文原文
    texts = [c["text"] for c in list_of_chunk]

    #得到每個chunk的embeddings
    embeddings = embed_chunk(texts)

    #把文件跟提取的向量放到向量數據庫中
    save_embeddings(list_of_chunk, embeddings)

if __name__ == "__main__":
    main()
