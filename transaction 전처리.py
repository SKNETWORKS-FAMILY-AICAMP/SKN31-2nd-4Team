import os
import gc
import pandas as pd
import numpy as np

########## 0. 경로 설정 ##########
VERSION = 1  # ← 1 또는 2로 바꾸면 v1/v2 전환

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_FILE  = os.path.join(RAW_DIR, f"transactions_v{VERSION}.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, f"transactions_v{VERSION}_processed.csv")

print(f"[INFO] 처리 버전  : v{VERSION}")
print(f"[INFO] 입력 파일  : {INPUT_FILE}")
print(f"[INFO] 출력 파일  : {OUTPUT_FILE}")

########## 1. 메모리 최적화 함수 ##########
def reduce_memory_usage_advanced(df):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object" or str(col_type).startswith("datetime"):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type).startswith("int"):
            if c_min >= -128 and c_max <= 127:
                df[col] = df[col].astype("int8")
            elif c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype("int16")
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df[col] = df[col].astype("int32")
            else:
                df[col] = df[col].astype("int64")
        elif str(col_type).startswith("float"):
            if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype("float16")
            elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype("float32")
            else:
                df[col] = df[col].astype("float64")

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"  메모리 최적화: {start_mem:.2f} MB → {end_mem:.2f} MB "
          f"({100 * (start_mem - end_mem) / start_mem:.1f}% 감소)")
    return df

########## 2. 데이터 로드 ##########
keep_cols = [
    'msno', 'payment_method_id', 'payment_plan_days',
    'plan_list_price', 'actual_amount_paid', 'is_auto_renew',
    'transaction_date', 'membership_expire_date', 'is_cancel'
]

print(f"\nv{VERSION} 데이터 로드 중...")
transactions = pd.read_csv(
    INPUT_FILE,
    usecols=keep_cols,
    dtype_backend="numpy_nullable"  # ArrowDtype 방지 (로컬 VS Code 호환)
)
# msno 명시적 str 변환 (groupby 속도 보장)
transactions["msno"] = transactions["msno"].astype(str)
transactions = reduce_memory_usage_advanced(transactions)
print(f"  shape: {transactions.shape}")

########## 3. 날짜 변환 ##########
print("\n날짜 컬럼 변환 중...")
transactions["transaction_date"] = pd.to_datetime(
    transactions["transaction_date"].astype(str), format="%Y%m%d", errors="coerce"
)
transactions["membership_expire_date"] = pd.to_datetime(
    transactions["membership_expire_date"].astype(str), format="%Y%m%d", errors="coerce"
)

# 이상치 날짜 제거 (9999년 등)
transactions = transactions[
    transactions["membership_expire_date"].dt.year <= 2030
]
print(f"  날짜 이상치 제거 후 shape: {transactions.shape}")

########## 4. 날짜 파생변수 ##########
print("날짜 기반 파생변수 생성 중...")
for prefix, col in [("trans", "transaction_date"), ("expire", "membership_expire_date")]:
    dt = transactions[col].dt
    transactions[f"{prefix}_year"]    = dt.year.astype("int16")
    transactions[f"{prefix}_month"]   = dt.month.astype("int8")
    transactions[f"{prefix}_day"]     = dt.day.astype("int8")
    transactions[f"{prefix}_weekday"] = dt.weekday.astype("int8")
print("  완료")

########## 5. 결제 파생변수 ##########
print("결제 파생변수 생성 중...")
transactions["pm_id_41"]   = (transactions["payment_method_id"] == 41).astype(int)
transactions["pm_id_38"]   = (transactions["payment_method_id"] == 38).astype(int)
transactions["pp_days_30"] = (transactions["payment_plan_days"] == 30).astype(int)

########## 6. 유저별 총 거래 횟수 (중복 제거 전) ##########
print("유저별 총 거래 횟수 계산 중...")
trans_count = transactions.groupby("msno").size().reset_index(name="trans_count")

########## 7. 유저별 최신 거래 1건만 남기기 ##########
print("유저별 최신 거래 내역 추출 중...")
transactions = (
    transactions
    .sort_values("transaction_date")
    .drop_duplicates("msno", keep="last")
)

########## 8. trans_count 병합 ##########
transactions = transactions.merge(trans_count, on="msno", how="left")

########## 9. 결측치 처리 ##########
numeric_cols = transactions.select_dtypes(include="number").columns
transactions[numeric_cols] = transactions[numeric_cols].fillna(transactions[numeric_cols].mean())

########## 10. 최종 메모리 최적화 ##########
print("\n최종 메모리 최적화 적용 중...")
transactions = reduce_memory_usage_advanced(transactions)
print(f"최종 shape: {transactions.shape}")

########## 11. 저장 ##########
transactions.to_csv(OUTPUT_FILE, index=False)
print(f"\n전처리 완료! 저장 경로: {OUTPUT_FILE}")

gc.collect()