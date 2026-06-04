import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ──────────────────────────────────────────────
# XGBoost 하이퍼파라미터 튜닝 근거
#
# max_depth        : 트리 깊이. 깊을수록 복잡한 패턴 학습 but 과적합 위험
#                    5~7이 이탈 예측 태스크에서 일반적으로 적합
# learning_rate    : 학습률. 낮을수록 안정적이나 n_estimators 많이 필요
# subsample        : 매 트리마다 사용할 행 비율 → 과적합 방지
# colsample_bytree : 매 트리마다 사용할 피처 비율 → 다양성 확보
# scale_pos_weight : 유지/이탈 비율(≈9.57) → 소수 클래스 가중치 부여
#
# n_estimators는 GridSearch 범위에서 제외하고 충분히 크게 고정
# (early_stopping 없이 GridSearchCV 사용하므로 300으로 고정)
# ──────────────────────────────────────────────

PARAM_GRID = {
    'max_depth'        : [4, 5, 6],
    'learning_rate'    : [0.05, 0.1],
    'subsample'        : [0.7, 0.8],
    'colsample_bytree' : [0.7, 0.8],
}


def tune_xgb(X_train, y_train, scale_pos_weight: float, cv: int = 5) -> dict:
    """
    GridSearchCV로 XGBoost 최적 하이퍼파라미터 탐색

    Parameters
    ----------
    scale_pos_weight : 유지(0) 수 / 이탈(1) 수  (data_loader.get_scale_pos_weight())
    cv               : StratifiedKFold 폴드 수

    Returns
    -------
    best_params : dict
    """
    print("########## XGBoost GridSearchCV 시작 ##########")
    t0 = time.time()

    base_model = XGBClassifier(
        n_estimators=300,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,  # 클래스 불균형 대응
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\n최적 파라미터 : {grid.best_params_}")
    print(f"최적 AUC(CV) : {grid.best_score_:.4f}")
    print(f"소요 시간    : {time.time() - t0:.1f}s")

    return grid.best_params_


def train_xgb(X_train, y_train, X_val, y_val, params: dict, scale_pos_weight: float):
    """
    최적 파라미터로 XGBoost 학습
    val 세트로 과적합 모니터링 (early_stopping_rounds=30)

    Parameters
    ----------
    params           : tune_xgb()에서 반환된 best_params
    scale_pos_weight : 클래스 가중치

    Returns
    -------
    model : 학습된 XGBClassifier
    """
    print("\n########## XGBoost 학습 ##########")
    t0 = time.time()

    model = XGBClassifier(
        **params,
        n_estimators=1000,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,           # val logloss 기준 조기 종료
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    print(f"최적 트리 수 : {model.best_iteration}")
    print(f"학습 완료   : {time.time() - t0:.1f}s")
    return model


def save_feature_importance(model, feature_names, results_dir: str = './results', top_n: int = 20):
    """
    Feature Importance 상위 top_n개를 CSV + PNG로 저장
    """
    os.makedirs(results_dir, exist_ok=True)

    imp_df = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    imp_df.to_csv(os.path.join(results_dir, 'xgb_feature_importance.csv'),
                  index=False, encoding='utf-8-sig')

    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['Feature'], top['Importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'XGBoost — Feature Importance Top {top_n}')
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'xgb_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [저장] xgb_feature_importance.csv / xgb_feature_importance.png")
    return imp_df
