"""
ColorBench 데이터셋 이상 샘플 탐지 스크립트
실행: python inspect_dataset.py
"""
import re
import ast
from collections import Counter
from datasets import load_dataset

ds = load_dataset("umd-zhou-lab/ColorBench", split="test")

def parse_choices(choices):
    if isinstance(choices, list):
        return choices
    if isinstance(choices, str):
        try:
            return ast.literal_eval(choices)
        except:
            return []
    return []

# ====================================================
# 1. answer 필드 전체 분포
# ====================================================
print("\n" + "="*60)
print("1. answer 필드 유니크 값 분포")
print("="*60)
ans_counter = Counter(str(row.get("answer","")).strip() for row in ds)
for val, cnt in sorted(ans_counter.items()):
    print(f"  '{val}' : {cnt}개")

# ====================================================
# 2. (E) 이상 — choices 범위 초과 샘플
# ====================================================
print("\n" + "="*60)
print("2. choices 범위 초과 샘플 (데이터 오염 의심)")
print("="*60)
overflow = []
for row in ds:
    ans = re.sub(r"[()]", "", str(row.get("answer",""))).strip().upper()
    choices = parse_choices(row.get("choices", []))
    if re.match(r"^[A-Z]$", ans):
        idx = ord(ans) - ord("A")
        if idx >= len(choices):
            overflow.append({
                "task":     row.get("task"),
                "question": row.get("question","")[:70],
                "answer":   row.get("answer"),
                "choices":  choices,
            })

print(f"범위 초과 샘플 수: {len(overflow)}개")
for s in overflow:
    print(f"\n  task    : {s['task']}")
    print(f"  question: {s['question']}")
    print(f"  answer  : {s['answer']}")
    print(f"  choices : {s['choices']}")

# ====================================================
# 3. choices가 아예 없는 샘플
# ====================================================
print("\n" + "="*60)
print("3. choices 필드가 비어있는 샘플")
print("="*60)
no_choices = []
for row in ds:
    choices = parse_choices(row.get("choices", []))
    if not choices:
        no_choices.append({
            "task":    row.get("task"),
            "answer":  row.get("answer"),
            "question":row.get("question","")[:70],
        })
print(f"choices 없는 샘플 수: {len(no_choices)}개")
for s in no_choices[:5]:
    print(f"  task={s['task']} | answer={s['answer']} | Q={s['question']}")

# ====================================================
# 4. 태스크별 샘플 수 & answer 형식 분포
# ====================================================
print("\n" + "="*60)
print("4. 태스크별 샘플 수 및 answer 형식")
print("="*60)
from collections import defaultdict
task_info = defaultdict(lambda: {"total": 0, "mcq": 0, "free": 0, "overflow": 0})
for row in ds:
    task = row.get("task", "Unknown")
    ans  = re.sub(r"[()]", "", str(row.get("answer",""))).strip().upper()
    choices = parse_choices(row.get("choices", []))
    task_info[task]["total"] += 1
    if re.match(r"^[A-D]$", ans):
        idx = ord(ans) - ord("A")
        if idx < len(choices):
            task_info[task]["mcq"] += 1
        else:
            task_info[task]["overflow"] += 1
    else:
        task_info[task]["free"] += 1

print(f"  {'Task':<22} {'Total':>6} {'MCQ':>6} {'Free':>6} {'Overflow':>9}")
print("  " + "-"*55)
for task, info in sorted(task_info.items()):
    print(f"  {task:<22} {info['total']:>6} {info['mcq']:>6} {info['free']:>6} {info['overflow']:>9}")

print("\n완료!")