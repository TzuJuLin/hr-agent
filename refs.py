"""條文引用解析：找出一條條文引用了哪些條文。

HR-07 第19條「依工作規則（HR-01）第17條至第22條…依勞基法第12條…」
  → 內部引用：HR-01 第17～22條
  → 外部法規：勞基法第12條（資料庫沒有收錄，只標註不抓取）
HR-02 第3條「依第4條」→ 同一份規章的 HR-02 第4條
「依 HR-03 第2章」這種章層級的引用不處理。
"""
import re


ARTICLE_REF = re.compile(
    r"第\s*(\d+(?:\s*[、,，及和]\s*\d+)*)\s*條"
    r"(?:\s*(?:至|到|~|～|－|-)\s*第?\s*(\d+)\s*條)?"
)
ARTICLE_HEADING = re.compile(r"^#+\s*第\s*(\d+)\s*條")

DOC_CODE_BEFORE = re.compile(r"(HR-\d+)\D{0,3}$")                 # 「工作規則（HR-01）」「HR-03 」
SELF_BEFORE = re.compile(r"本\S{0,3}(?:法|則|規章|辦法)\s*$")       # 「本辦法」「本規則」→ 同一份規章
LAW_BEFORE = re.compile(r"([一-鿿]*(?:法|條例|細則|規則|辦法|準則))\s*$")
LEADING_VERB = re.compile(r"^.*(依照|依據|根據|參照|依|按|及|與|和|條(?!例))")  # 「應依性別平等工作法」→「性別平等工作法」


def article_number(text: str) -> int | None:
    m = ARTICLE_HEADING.match(text.strip())
    return int(m.group(1)) if m else None


def parse_references(text: str, current_doc: str, titles: dict[str, str]) -> tuple[list[tuple[str, int]], list[str]]:
    """回傳 (內部引用 [(規章代號, 條號), ...], 外部法規 ["勞基法第12條", ...])。

    titles：內部規章名稱 → 代號，例如 {"工作規則": "HR-01"}，
    讓沒寫代號的「依差勤管理辦法第8條」也能對應到內部規章，而不是被當成外部法規。
    """
    lines = text.strip().splitlines()
    self_no = article_number(text)
    body = "\n".join(lines[1:]) if self_no is not None else text   # 跳過標題行

    internal, external = [], []
    for match in ARTICLE_REF.finditer(body):
        before = body[max(0, match.start() - 20):match.start()]
        nums = [int(n) for n in re.split(r"[、,，及和\s]+", match.group(1)) if n]
        if match.group(2):                                        
            nums = list(range(nums[0], int(match.group(2)) + 1))

        doc = None
        code = DOC_CODE_BEFORE.search(before)
        law = LAW_BEFORE.search(before)
        if code:
            doc = code.group(1)
        elif SELF_BEFORE.search(before):
            doc = current_doc
        elif law:
            name = LEADING_VERB.sub("", law.group(1))
            doc = next((c for t, c in titles.items() if name and (name in t or t in name)), None)
            if doc is None:                                          # 外部法規：只記錄
                ref = name + re.sub(r"\s+", "", match.group(0))
                if ref not in external:
                    external.append(ref)
                continue
        else:
            doc = current_doc

        for n in nums:
            ref = (doc, n)
            if ref != (current_doc, self_no) and ref not in internal:
                internal.append(ref)
    return internal, external
