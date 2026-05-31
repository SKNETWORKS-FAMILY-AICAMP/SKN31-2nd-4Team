# =============================================================
# preprocess.py — 공통 전처리 & 데이터 로딩
# =============================================================
# 각 모델 학습 파일에서 import해서 사용합니다.
#   from preprocess import load_and_preprocess
# =============================================================

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)


def load_and_preprocess(data_path='./data/train_features.csv', test_size=0.2, random_state=42):
    """
    train_features.csv를 로딩하고 전처리 후
    X_train, X_val, y_train, y_val, feature_cols 반환
    """

    print("▶ 데이터 로딩 중...")
    df = pd.read_csv(data_path)
    print(f"  shape: {df.shape}")

    # ── 사용하지 않을 컬럼 제거 ──────────────────────────────
    DROP_COLS = [
        'msno', 'is_churn',
        'last_tx_date', 'last_expire_date', 'last_listen_date',
        'age_bucket', 'tenure_bucket',      # 범주형 구간 (수치 피처로 대체)
    ]
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X = df[feature_cols].copy()
    y = df['is_churn'].copy()

    print(f"  피처 수    : {len(feature_cols)}")
    print(f"  이탈률     : {y.mean():.4f} ({y.mean()*100:.2f}%)")

    # ── 결측치 처리 ──────────────────────────────────────────
    # 범주형(object) → LabelEncoder, 수치형 → 중앙값
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            X[col] = X[col].fillna(X[col].median())

    # ── Train / Validation Split ─────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape  : {X_val.shape}\n")

    return X_train, X_val, y_train, y_val, feature_cols