import pickle
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# 모델 학습에서 제외할 컬럼
#   - msno                  : 회원 ID (식별자)
#   - registration_init_time: 날짜 문자열 원본
#   - transaction_date      : 날짜 문자열 원본
#   - membership_expire_date: 날짜 문자열 원본
#   - last_listen_date      : 날짜 정수값이나 0값 다수 + 누수 위험
# ──────────────────────────────────────────────
DROP_COLS = [
    'registration_init_time',
    'transaction_date',
    'membership_expire_date',
    'last_listen_date',
    'expire_month',    # 데이터 누수: 타깃 정의(4월 갱신안함)과 직접 연결
    'expire_year',     # 데이터 누수: 2017년 3월 기준 데이터라 2017년에 편중
    'trans_month',     # 데이터 누수: 3월 거래 기준 데이터라 3월에 편중
    'trans_year',      # 데이터 누수: 2017년 거래 기준으로 2017년에 편중
]

TARGET = 'is_churn'


def load_data(data_dir: str = './data'):
    """
    train/val/test pkl 파일을 로드하고 X, y로 분리해 반환

    Parameters
    ----------
    data_dir : str
        pkl 파일이 위치한 디렉토리 경로

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : pd.DataFrame / pd.Series
    """
    data_dir = Path(data_dir)

    print("데이터 로드 중...")
    with open(data_dir / 'train_split.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(data_dir / 'val_split.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(data_dir / 'test_split.pkl', 'rb') as f:
        test = pickle.load(f)

    print(f"  train : {train.shape}")
    print(f"  val   : {val.shape}")
    print(f"  test  : {test.shape}")

    # ── 클래스 분포 출력 ──────────────────────────────────
    print("\n[클래스 분포]")
    for name, df in [('train', train), ('val', val), ('test', test)]:
        n1 = df[TARGET].sum()
        n0 = len(df) - n1
        print(f"  {name}: 유지={n0:,}  이탈={n1:,}  이탈율={n1/len(df)*100:.2f}%")

    # ── X / y 분리 ────────────────────────────────────────
    drop_train = [c for c in DROP_COLS if c in train.columns]
    drop_val   = [c for c in DROP_COLS if c in val.columns]
    drop_test  = [c for c in DROP_COLS if c in test.columns]

    X_train = train.drop(columns=drop_train + [TARGET]).set_index('msno')
    y_train = train[TARGET]

    X_val   = val.drop(columns=drop_val + [TARGET]).set_index('msno')
    y_val   = val[TARGET]

    X_test  = test.drop(columns=drop_test + [TARGET]).set_index('msno')
    y_test  = test[TARGET]

    print(f"\n사용 피처 수: {X_train.shape[1]}개")
    print(f"피처 목록: {list(X_train.columns)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_scale_pos_weight(y_train: pd.Series) -> float:
    """
    XGBoost / LightGBM 클래스 불균형 대응용 가중치 계산
    scale_pos_weight = 유지(0) 수 / 이탈(1) 수
    """
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    return round(n0 / n1, 4)
