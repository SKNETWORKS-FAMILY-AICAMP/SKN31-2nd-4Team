"""
user_logs_v2.csv → msno 기준 groupby 통계 집계
출력: data/processed/user_logs_aggregated.csv
"""

import os
import time
import pandas as pd

# ── 경로 설정 ──────────────────────────────────────────────
RAW_PATH  = "data/raw/user_logs_v2.csv"
OUT_DIR   = "data/processed"
OUT_PATH  = os.path.join(OUT_DIR, "user_logs_aggregated.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 청크 기반 로드 (대용량 파일 대응) ──────────────────────
CHUNK_SIZE = 500_000

print(f"[1] 파일 로드 시작: {RAW_PATH}")
t0 = time.time()

chunks = []
for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE)):
    chunks.append(chunk)
    print(f"    chunk {i+1} 로드 완료 ({len(chunk):,} rows)")

df = pd.concat(chunks, ignore_index=True)
print(f"[1] 로드 완료 — 총 {len(df):,} rows | {time.time()-t0:.1f}s")

# ── date 컬럼 정수 처리 (max 연산용) ───────────────────────
print("[2] 전처리 시작")
t1 = time.time()
df["date"] = pd.to_numeric(df["date"], errors="coerce")
print(f"[2] 전처리 완료 | {time.time()-t1:.1f}s")

# ── msno 기준 groupby 집계 ──────────────────────────────────
print("[3] groupby 집계 시작")
t2 = time.time()

agg_df = df.groupby("msno", sort=False).agg(
    total_secs       = ("total_secs", "sum"),
    num_25           = ("num_25",     "sum"),
    num_50           = ("num_50",     "sum"),
    num_75           = ("num_75",     "sum"),
    num_985          = ("num_985",    "sum"),
    num_100          = ("num_100",    "sum"),
    num_unq          = ("num_unq",    "sum"),
    log_count        = ("date",       "count"),
    last_listen_date = ("date",       "max"),
).reset_index()

print(f"[3] 집계 완료 — 유저 수: {len(agg_df):,} | {time.time()-t2:.1f}s")

# ── 저장 ───────────────────────────────────────────────────
print(f"[4] 저장 중: {OUT_PATH}")
t3 = time.time()
agg_df.to_csv(OUT_PATH, index=False)
print(f"[4] 저장 완료 | {time.time()-t3:.1f}s")

# ── 결과 미리보기 ───────────────────────────────────────────
print("\n========== 결과 미리보기 ==========")
print(f"shape : {agg_df.shape}")
print(f"dtypes:\n{agg_df.dtypes}")
print(f"\n{agg_df.head(3).to_string()}")

total_elapsed = time.time() - t0
print(f"\n전체 소요 시간: {total_elapsed:.1f}s")
print("Done.")