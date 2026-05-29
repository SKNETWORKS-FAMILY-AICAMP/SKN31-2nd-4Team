# =============================================================
# KKBox 고객이탈 예측 - 피처 엔지니어링 전체 파이프라인
# =============================================================

import pandas as pd
import numpy as np


# =============================================================
# STEP 1. CSV 파일 로딩
# =============================================================

print("▶ 데이터 로딩 시작...")

train        = pd.read_csv('./data/train_v2.csv')
members      = pd.read_csv('./data/members_v3.csv')
transactions = pd.read_csv('./data/transactions_v2.csv')
# user_logs    = pd.read_csv('./data/user_logs_agg.csv')  # 가공된 파일 로딩

# user_logs는 용량이 크므로 chunk 단위로 로딩
print("▶ user_logs 로딩 중 (시간이 걸릴 수 있어요)...")
# chunk_list = []
# for chunk in pd.read_csv('./data/user_logs_v2.csv', chunksize=500_000):
#     chunk_list.append(chunk)
# user_logs = pd.concat(chunk_list, ignore_index=True)

user_logs = pd.read_csv('./data/user_logs_agg.csv')

print(f"  train        : {train.shape}")
print(f"  members      : {members.shape}")
print(f"  transactions : {transactions.shape}")
print(f"  user_logs    : {user_logs.shape}")
print("▶ 로딩 완료!\n")


# =============================================================
# STEP 2. 함수 정의
# =============================================================

REF_DATE = pd.Timestamp('2017-03-01')  # 예측 기준일


# ── 2-1. 회원 인구통계 피처 ──────────────────────────────────
def build_member_features(members):
    df = members.copy()

    # 가입 후 경과일
    df['registration_date'] = pd.to_datetime(
        df['registration_init_time'], format='%Y%m%d', errors='coerce'
    )
    df['tenure_days'] = (REF_DATE - df['registration_date']).dt.days

    # 나이 이상치 처리 (0~100 범위 외 → NaN)
    df['bd'] = df['bd'].where((df['bd'] > 0) & (df['bd'] < 100), np.nan)
    df['age_bucket'] = pd.cut(
        df['bd'],
        bins=[0, 18, 25, 35, 45, 60, 100],
        labels=['~18', '19-25', '26-35', '36-45', '46-60', '60+']
    )

    # 성별 인코딩
    df['gender_enc'] = df['gender'].map({'male': 0, 'female': 1}).fillna(-1)

    # tenure 구간
    df['tenure_bucket'] = pd.cut(
        df['tenure_days'],
        bins=[-1, 90, 365, 730, 9999],
        labels=['신규', '중기', '장기', '초장기']
    )

    return df[['msno', 'tenure_days', 'bd', 'age_bucket',
               'gender_enc', 'city', 'registered_via', 'tenure_bucket']]


# ── 2-2. 결제 행동 피처 ──────────────────────────────────────
def build_transaction_features(transactions):
    df = transactions.copy()
    df['transaction_date']      = pd.to_datetime(df['transaction_date'],      format='%Y%m%d')
    df['membership_expire_date'] = pd.to_datetime(df['membership_expire_date'], format='%Y%m%d')

    # 할인율
    df['discount_rate'] = 1 - (
        df['actual_amount_paid'] / df['plan_list_price'].replace(0, np.nan)
    )
    df['is_discounted'] = (df['discount_rate'] > 0.05).astype(int)

    agg = df.groupby('msno').agg(
        tx_count          = ('transaction_date',      'count'),
        avg_paid          = ('actual_amount_paid',    'mean'),
        avg_discount_rate = ('discount_rate',         'mean'),
        discount_ratio    = ('is_discounted',         'mean'),   # 할인 의존도
        auto_renew_ratio  = ('is_auto_renew',         'mean'),
        cancel_ratio      = ('is_cancel',             'mean'),
        plan_days_mean    = ('payment_plan_days',     'mean'),
        plan_variety      = ('payment_plan_days',     'nunique'),
        payment_methods   = ('payment_method_id',    'nunique'),
        last_tx_date      = ('transaction_date',      'max'),
        last_expire_date  = ('membership_expire_date','max'),
    ).reset_index()

    # 최근 결제 경과일 & 만료까지 남은 일수
    agg['days_since_last_tx'] = (REF_DATE - agg['last_tx_date']).dt.days
    agg['days_until_expire']  = (agg['last_expire_date'] - REF_DATE).dt.days

    # 플랜 다운그레이드 횟수
    df_sorted = df.sort_values(['msno', 'transaction_date'])
    df_sorted['plan_change'] = df_sorted.groupby('msno')['payment_plan_days'].diff().fillna(0)
    plan_chg = (
        df_sorted.groupby('msno')['plan_change']
        .apply(lambda x: (x < 0).sum())
        .reset_index(name='downgrade_count')
    )

    agg = agg.merge(plan_chg, on='msno', how='left')
    return agg


# ── 2-3. 청취 행동 피처 ──────────────────────────────────────
def build_userlog_features(user_logs):

   df = user_logs.copy()
   
   total = (df['num_25'] + df['num_50'] + df['num_75']+ df['num_985'] + df['num_unq'])
   df['completion_rate'] = (df['num_985'] + df['num_unq']) / total.replace(0, np.nan)
   df['skip_rate']       = df['num_25'] / total.replace(0, np.nan)
   df['listen_regularity'] = df['num_unq'] / df['log_count'].replace(0, np.nan)  # std 대신 1인당 평균으로 대체
   
   df['days_since_last_listen'] = (REF_DATE - pd.to_datetime(df['last_listen_date'])).dt.days
   df['listen_trend_ratio'] = df['plays_recent30'] / df['plays_prev30'].replace(0, np.nan)
   
   return df[['msno', 'total_secs', 'num_unq', 'log_count', 'active_days',
               'completion_rate', 'skip_rate', 'listen_regularity',
               'last_listen_date', 'days_since_last_listen',
               'plays_recent30', 'plays_prev30', 'listen_trend_ratio']]
    # df = user_logs.copy()
    # df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    # # 완료율 & 스킵률
    # total = (df['num_25'] + df['num_50'] + df['num_75']
    #          + df['num_985'] + df['num_unq'])
    # df['completion_rate'] = (df['num_985'] + df['num_unq']) / total.replace(0, np.nan)
    # df['skip_rate']       = df['num_25'] / total.replace(0, np.nan)

    # agg = df.groupby('msno').agg(
    #     total_plays       = ('num_unq',        'sum'),
    #     total_secs        = ('total_secs',       'sum'),
    #     active_days       = ('date',             'nunique'),
    #     avg_daily_plays   = ('num_unq',         'mean'),
    #     avg_completion    = ('completion_rate',  'mean'),
    #     avg_skip          = ('skip_rate',        'mean'),
    #     listen_regularity = ('num_unq',         'std'),   # 낮을수록 규칙적
    #     last_listen_date  = ('date',             'max'),
    # ).reset_index()

    # agg['days_since_last_listen'] = (REF_DATE - agg['last_listen_date']).dt.days

    # # 최근 30일 vs 이전 30일 청취량 트렌드
    # recent = df[df['date'] >= REF_DATE - pd.Timedelta(days=30)]
    # prev   = df[
    #     (df['date'] >= REF_DATE - pd.Timedelta(days=60)) &
    #     (df['date'] <  REF_DATE - pd.Timedelta(days=30))
    # ]
    # recent_agg = recent.groupby('msno')['num_unq'].sum().reset_index(name='plays_recent30')
    # prev_agg   = prev.groupby('msno')['num_unq'].sum().reset_index(name='plays_prev30')

    # agg = agg.merge(recent_agg, on='msno', how='left')
    # agg = agg.merge(prev_agg,   on='msno', how='left')

    # # 트렌드 비율 (1 미만 = 청취량 감소 → 이탈 위험)
    # agg['listen_trend_ratio'] = (
    #     agg['plays_recent30'] / agg['plays_prev30'].replace(0, np.nan)
    # )

    # return agg


# ── 2-4. 최종 데이터셋 통합 ──────────────────────────────────
def build_final_dataset(train, members, transactions, user_logs):
    member_feat = build_member_features(members)
    tx_feat     = build_transaction_features(transactions)
    log_feat    = build_userlog_features(user_logs)

    df = train.merge(member_feat, on='msno', how='left')
    df = df.merge(tx_feat,        on='msno', how='left')
    df = df.merge(log_feat,       on='msno', how='left')

    # avg_completion → completion_rate, avg_skip → skip_rate 로 변경
    df['engagement_score'] = (
        df['completion_rate'].fillna(0) * 0.4 +
        (1 - df['skip_rate'].fillna(0)) * 0.3 +
        (df['active_days'] / df['plan_days_mean'].clip(1)).fillna(0) * 0.3
    )

    return df


# =============================================================
# STEP 3. 실행
# =============================================================

print("▶ 피처 엔지니어링 시작...")
df = build_final_dataset(train, members, transactions, user_logs)

print(f"▶ 완료! 최종 데이터셋: {df.shape}")
print(df.head())


# =============================================================
# STEP 4. 결과 저장 (선택)
# =============================================================

df.to_csv('./data/train_features.csv', index=False)
print("▶ 저장 완료: ./data/train_features.csv")
