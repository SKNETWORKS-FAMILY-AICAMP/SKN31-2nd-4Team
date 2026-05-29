import pandas as pd

REF_DATE = pd.Timestamp('2017-03-01')

print("▶ train 유저 ID 로딩 중...")
train = pd.read_csv('./data/train_v2.csv')
target_msno = set(train['msno'])
print(f"  대상 유저 수: {len(target_msno):,}명")

print("▶ user_logs 로딩 및 집계 중 (시간이 걸릴 수 있어요)...")

agg_list    = []
recent_list = []
prev_list   = []
active_list = []

for chunk in pd.read_csv('./data/user_logs.csv', chunksize=500_000):
    chunk = chunk[chunk['msno'].isin(target_msno)]
    if chunk.empty:
        continue

    chunk['date'] = pd.to_datetime(chunk['date'], format='%Y%m%d')

    # ① 전체 집계
    agg = chunk.groupby('msno').agg(
        total_secs       = ('total_secs', 'sum'),
        num_25           = ('num_25',     'sum'),
        num_50           = ('num_50',     'sum'),
        num_75           = ('num_75',     'sum'),
        num_985          = ('num_985',    'sum'),
        num_100          = ('num_100',    'sum'),
        num_unq          = ('num_unq',    'sum'),
        log_count        = ('date',       'count'),
        last_listen_date = ('date',       'max'),
    ).reset_index()
    agg_list.append(agg)

    # ② 최근 30일 / 이전 30일 raw 행 보존
    recent_list.append(
        chunk[chunk['date'] >= REF_DATE - pd.Timedelta(days=30)][['msno', 'num_unq']]
    )
    prev_list.append(
        chunk[
            (chunk['date'] >= REF_DATE - pd.Timedelta(days=60)) &
            (chunk['date'] <  REF_DATE - pd.Timedelta(days=30))
        ][['msno', 'num_unq']]
    )

    # ③ active_days용 날짜 보존
    active_list.append(chunk[['msno', 'date']])

# ④ 전체 합산
print("▶ 최종 집계 중...")
user_logs = (
    pd.concat(agg_list, ignore_index=True)
      .groupby('msno', as_index=False)
      .agg({
          'total_secs'      : 'sum',
          'num_25'          : 'sum',
          'num_50'          : 'sum',
          'num_75'          : 'sum',
          'num_985'         : 'sum',
          'num_100'         : 'sum',
          'num_unq'         : 'sum',
          'log_count'       : 'sum',
          'last_listen_date': 'max',
      })
)

# ⑤ active_days 계산
active_agg = (
    pd.concat(active_list, ignore_index=True)
      .groupby('msno')['date']
      .nunique()
      .reset_index(name='active_days')
)

# ⑥ 트렌드 계산
recent_agg = (
    pd.concat(recent_list, ignore_index=True)
      .groupby('msno')['num_unq']
      .sum()
      .reset_index(name='plays_recent30')
)
prev_agg = (
    pd.concat(prev_list, ignore_index=True)
      .groupby('msno')['num_unq']
      .sum()
      .reset_index(name='plays_prev30')
)

# ⑦ 최종 merge
user_logs = user_logs.merge(active_agg,  on='msno', how='left')
user_logs = user_logs.merge(recent_agg,  on='msno', how='left')
user_logs = user_logs.merge(prev_agg,    on='msno', how='left')

user_logs.to_csv('./data/user_logs_agg.csv', index=False)
print(f"✅ 저장 완료: ./data/user_logs_agg.csv")
print(f"   shape: {user_logs.shape}")
print(f"   columns: {user_logs.columns.tolist()}")