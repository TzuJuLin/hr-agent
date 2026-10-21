import json
import re
from datetime import datetime
from pathlib import Path
 
import pandas as pd
 
from rag import rerank, retrieve
 
PROJECT_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = PROJECT_DIR / "questions.json"
OUTPUT_DIR = PROJECT_DIR / "eval_results"
 
TOP_K_RETRIEVED = 20          # retrieve 撈幾個候選
RETRIEVE_KS = [5, 10, 20]     # 在 retrieve 結果上計算 recall@k
RERANK_KS = [1, 3, 5]         # 在 rerank 結果上計算 recall@k

def metric_name(stage: str, k: int) -> str:
    return f"{stage}_{'hit' if k == 1 else 'recall'}@{k}"
 
# ---------- 判斷與計算 ----------
 
def is_article(text: str, source: str, doc: str, article: int) -> bool:
    """chunk 是否為指定規章的指定條文。
    因為metadata中只有相關文件沒說在第幾條
    只比對 chunk 開頭的標題（例如「### 第3條（…）」），
   
    """
    #如果文章不對直接return False
    if not source.startswith(doc):
        return False
    #判斷是否是正確的條文
    return re.match(rf"#+\s*第\s*{article}\s*條", text.strip()) is not None
 
 
def find_rank(chunks: list[tuple[str, str]], doc: str, article: int) -> int | None:
    """回傳條文在清單中的名次（從 1 開始），找不到回傳 None。"""
    for rank, (text, source) in enumerate(chunks, start=1):
        #如果有找到要找的條文，return排名
        if is_article(text, source, doc, article):
            return rank
    return None
 


def recall_at(ranks: list[int | None], k: int) -> float:
    """相關條文中，有幾成排在前 k 名。"""
    #ranks:相關條文的名次
    return sum(r is not None and r <= k for r in ranks) / min(len(ranks), k)
 
 
def all_found_at(ranks: list[int | None], k: int) -> bool:
    """相關條文是否全部排在前 k 名（跨文件題最重要的指標）。"""
    return all(r is not None and r <= k for r in ranks)
 


def first_rank(ranks: list[int | None]) -> int | None:
    """相關條文中名次最前面的是第幾名"""
    found = [r for r in ranks if r is not None]
    return min(found) if found else None
 
 
def chunk_label(text: str, source: str) -> str:
    """看模型到底撈到哪個條文。把 chunk 簡化成「HR-04 第11條（特休天數）」方便閱讀。"""
    heading = text.strip().splitlines()[0].lstrip("#").strip()
    return f"{source.split('_')[0]} {heading[:25]}"
 
 
def top_labels(chunks: list[tuple[str, str]], n: int = 3) -> str:
    return "\n".join(f"{i}. {chunk_label(t, s)}" for i, (t, s) in enumerate(chunks[:n], start=1))
 
 
# ---------- 單題評估 ----------
 
def evaluate_question(q: dict) -> tuple[dict, list[dict]]:
    retrieved = retrieve(q["question"], TOP_K_RETRIEVED)
    # 對全部候選重排，才能知道每個條文重排後的完整名次
    reranked = rerank(query=q["question"], retrieved_chunks=retrieved, top_k=len(retrieved))
 
    article_rows = []
    retrieve_ranks, rerank_ranks = [], []
    for ref in q["relevant_articles"]:
        #r1, r2是回傳的名次
        r1 = find_rank(retrieved, ref["doc"], ref["article"])
        r2 = find_rank(reranked, ref["doc"], ref["article"])
        retrieve_ranks.append(r1)
        rerank_ranks.append(r2)
        article_rows.append({
            "題號": q["id"],
            "類別": q["category"],
            "規章": ref["doc"],
            "條號": ref["article"],
            "retrieve名次": r1,
            "rerank名次": r2,
            "名次變化": (r1 - r2) if (r1 and r2) else None,  # 正數 = reranker 把它往前排
        })
 
    row = {
        "題號": q["id"],
        "類別": q["category"],
        "問題": q["question"],
        "相關條文數": len(q["relevant_articles"]),
    }
    """把名次換算成各種分書"""
    if retrieve_ranks:
        for k in RETRIEVE_KS:
            row[metric_name("retrieve", k)] = recall_at(retrieve_ranks, k)
        row[f"retrieve_all_found@{TOP_K_RETRIEVED}"] = all_found_at(retrieve_ranks, TOP_K_RETRIEVED)
        row["retrieve_first_rank"] = first_rank(retrieve_ranks)

        
        for k in RERANK_KS:
            row[metric_name("rerank", k)] = recall_at(rerank_ranks, k)
        #如果k<relevant articles則不計算，回傳None
        for k in RERANK_KS:
            row[f"rerank_all_found@{k}"] = all_found_at(rerank_ranks, k) if k >= row["相關條文數"] else None
        row["rerank_first_rank"] = first_rank(rerank_ranks)
        fr = row["rerank_first_rank"]
        row["rerank_RR"] = 1 / fr if fr else 0.0  # 平均後即為 MRR
 
    row["retrieve前3名"] = top_labels(retrieved)
    row["rerank前3名"] = top_labels(reranked)
    return row, article_rows
 
 
# ---------- 匯總與輸出 ----------
 
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if "recall@" in c or "all_found@" in c or c == "rerank_RR" or "hit@" in c]
    scored = df[df["相關條文數"] > 0]
    scored = scored.astype({c: float for c in metric_cols})
 
    by_cat = scored.groupby("類別")[metric_cols].mean()
    by_cat.insert(0, "題數", scored.groupby("類別").size())
 
    overall = scored[metric_cols].mean().to_frame().T
    overall.insert(0, "題數", len(scored))
    overall.index = ["整體"]
 
    summary = pd.concat([overall, by_cat])
    summary = summary.rename(columns={"rerank_RR": "rerank_MRR"})
    summary.index.name = "類別"
    return summary.reset_index()
 
 
def format_workbook(writer: pd.ExcelWriter, sheets: dict[str, pd.DataFrame]) -> None:
    from openpyxl.styles import Alignment, Font
 
    for name, df in sheets.items():
        ws = writer.sheets[name]
        ws.freeze_panes = "B2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
 
        for idx, col in enumerate(df.columns, start=1):
            letter = ws.cell(row=1, column=idx).column_letter
            longest = max([len(str(col))] + [len(str(v).split("\n")[0]) for v in df[col]])
            ws.column_dimensions[letter].width = min(max(10, longest * 1.6), 45)
 
            is_pct = "recall@" in col or "all_found@" in col or "hit@" in col
            for cell in ws[letter][1:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if is_pct and isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                    cell.number_format = "0%"
                elif col == "rerank_MRR" and isinstance(cell.value, float):
                    cell.number_format = "0.00"
 
 
def main() -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
 
    question_rows, article_rows = [], []
    for q in questions:
        print(f"評估 {q['id']} …")
        row, arts = evaluate_question(q)
        question_rows.append(row)
        article_rows.extend(arts)
 
    df_questions = pd.DataFrame(question_rows)
    df_articles = pd.DataFrame(article_rows)
    df_summary = build_summary(df_questions)
    df_settings = pd.DataFrame([
        {"參數": "評估時間", "值": datetime.now().strftime("%Y-%m-%d %H:%M")},
        {"參數": "題目檔", "值": QUESTIONS_FILE.name},
        {"參數": "題數", "值": len(questions)},
        {"參數": "TOP_K_RETRIEVED", "值": TOP_K_RETRIEVED},
        {"參數": "RETRIEVE_KS", "值": str(RETRIEVE_KS)},
        {"參數": "RERANK_KS", "值": str(RERANK_KS)},
    ])
 
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"rank_eval_{datetime.now():%Y%m%d_%H%M}.xlsx"
    sheets = {
        "摘要": df_summary,
        "題目層級": df_questions,
        "條文層級": df_articles,
        "設定": df_settings,
    }
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
        format_workbook(writer, sheets)
 
    overall = df_summary.iloc[0]
    print("\n===== 整體結果 =====")
    for k in RETRIEVE_KS:
        col = metric_name("retrieve", k)
        print(f"{col}: {overall[col]:.0%}")
    for k in RERANK_KS:
        col = metric_name("rerank", k)
        print(f"{col}: {overall[col]:.0%}"
            f"   全部找到@{k}: {overall[f'rerank_all_found@{k}']:.0%}")
    print(f"rerank MRR: {overall['rerank_MRR']:.2f}")
 
    missing = df_articles[df_articles["retrieve名次"].isna()]
    if not missing.empty:
        print(f"\n前 {TOP_K_RETRIEVED} 名都沒找到的條文：")
        for _, m in missing.iterrows():
            print(f"  {m['題號']}  {m['規章']} 第{m['條號']}條")
 
    print(f"\n已輸出：{out_path}")
 
 
if __name__ == "__main__":
    main()
