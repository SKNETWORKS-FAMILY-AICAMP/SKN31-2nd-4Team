# =============================================================
# merge_datasets.py — 전처리된 파일 통합
# =============================================================
# 입력:
#   ./data/train_v2.csv
#   ./data/members_preprocessed.csv
#   ./data/transactions_v2_processed.csv
#
# 출력:
#   ./data/merged_dataset_v1.csv
# =============================================================

import pandas as pd

print("▶ 파일 로딩 중...")

train        = pd.read_csv('./data/train_v2.csv')
members      = pd.read_csv('./data/members_preprocessed.csv')
transactions = pd.read_csv('./data/transactions_v2_processed.csv')

print(f"  train_v2              : {train.shape}")
print(f"  members_preprocessed  : {members.shape}")
print(f"  transactions_processed: {transactions.shape}")


# =============================================================
# STEP 1. train 기준 msno 필터링
# =============================================================

train_msnos  = set(train['msno'])

members      = members[members['msno'].isin(train_msnos)]
transactions = transactions[transactions['msno'].isin(train_msnos)]

print(f"\n▶ train msno 기준 필터링 후")
print(f"  members      : {members.shape}")
print(f"  transactions : {transactions.shape}")


# =============================================================
# STEP 2. transactions 집계 (msno 단위)
# =============================================================
# transactions는 msno당 여러 행이 있으므로 집계 후 merge

print("\n▶ transactions 집계 중...")

# 수치형 컬럼만 자동 감지해서 집계
tx_numeric = transactions.select_dtypes(include='number').columns.tolist()
tx_numeric = [c for c in tx_numeric if c != 'msno']

# 집계 방식 정의 — 컬럼 성격에 따라 자동 분기
agg_dict = {}
for col in tx_numeric:
    if col in ['is_auto_renew', 'is_cancel']:
        agg_dict[col] = 'mean'       # 비율
    elif col in ['payment_plan_days', 'plan_list_price', 'actual_amount_paid']:
        agg_dict[col] = 'mean'       # 평균값
    elif col in ['payment_method_id']:
        agg_dict[col] = 'nunique'    # 다양성
    else:
        agg_dict[col] = 'mean'       # 나머지는 평균

# 날짜형 컬럼 처리
date_cols = [c for c in transactions.columns
             if 'date' in c.lower() and c != 'msno']
for col in date_cols:
    transactions[col] = pd.to_datetime(transactions[col], errors='coerce')
    agg_dict[col] = 'max'            # 가장 최근 날짜

tx_agg = transactions.groupby('msno').agg(agg_dict).reset_index()

# 컬럼명 중복 방지용 prefix
tx_agg.columns = [
    f'tx_{c}' if c != 'msno' else c
    for c in tx_agg.columns
]

print(f"  집계 완료: {tx_agg.shape}")


# =============================================================
# STEP 3. 병합 (train ← members ← transactions)
# =============================================================

print("\n▶ 데이터 병합 중...")

df = train.merge(members, on='msno', how='inner')
df = df.merge(tx_agg,    on='msno', how='inner')

print(f"  최종 shape : {df.shape}")
print(f"  컬럼 목록  : {df.columns.tolist()}")


# =============================================================
# STEP 4. 결측치 확인
# =============================================================

null_ratio = (df.isnull().sum() / len(df)).round(3)
null_cols  = null_ratio[null_ratio > 0]

if len(null_cols) > 0:
    print(f"\n▶ 결측치 비율 (0 초과 컬럼만)")
    print(null_cols.to_string())
else:
    print("\n▶ 결측치 없음 ✅")


# =============================================================
# STEP 5. 저장
# =============================================================

df.to_csv('./data/merged_dataset_v2.csv', index=False)
print(f"\n✅ 저장 완료: ./data/merged_dataset_v2.csv")
print(f"   shape  : {df.shape}")
print(f"   이탈률 : {df['is_churn'].mean():.4f} ({df['is_churn'].mean()*100:.2f}%)")