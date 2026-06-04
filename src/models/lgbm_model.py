import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from lightgbm import early_stopping, log_evaluation

# ──────────────────────────────────────────────
# LightGBM 하이퍼파라미터 튜닝 근거
#
# num_leaves       : 트리의 리프 수. 2^max_depth보다 유연하게 복잡도 제어
#                    클수록 복잡한 패턴 학습 but 과적합 위험
# learning_rate    : 학습률. 낮을수록 안정적이나 학습 느림
# min_child_samples: 리프 최소 샘플 수 → 작으면 노이즈 과적합
# subsample        : 행 샘플링 비율 → 다양성 확보 및 과적합 방지
# colsample_bytree : 피처 샘플링 비율
# scale_pos_weight : 유지/이탈 비율 → 소수 클래스 가중치 부여
#
# n_estimators는 충분히 크게 설정(2000) 후 early_stopping으로 최적 시점 탐색
# early_stopping_rounds=50 → val logloss 50 라운드 개선 없으면 조기 종료
# ──────────────────────────────────────────────

PARAM_GRID = {
    'num_leaves'       : [31, 63, 127],
    'learning_rate'    : [0.05, 0.1],
    'min_child_samples': [20, 50, 100],
    'subsample'        : [0.7, 0.8],
}


def tune_lgbm(X_train, y_train, scale_pos_weight: float, cv: int = 5) -> dict:
    """
    GridSearchCV로 LightGBM 최적 하이퍼파라미터 탐색
    (GridSearchCV 단계에서는 n_estimators 고정, early_stopping 미사용)

    Parameters
    ----------
    scale_pos_weight : 유지(0) 수 / 이탈(1) 수
    cv               : StratifiedKFold 폴드 수

    Returns
    -------
    best_params : dict
    """
    print("########## LightGBM GridSearchCV 시작 ##########")
    t0 = time.time()

    base_model = lgb.LGBMClassifier(
        n_estimators=300,
        objective='binary',
        metric='binary_logloss',
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # 클래스 불균형 대응
        random_state=42,
        n_jobs=-1,
        verbose=-1,
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


def train_lgbm(X_train, y_train, X_val, y_val, params: dict, scale_pos_weight: float):
    """
    최적 파라미터로 LightGBM 학습
    early_stopping으로 val logloss 기준 최적 트리 수 탐색

    Parameters
    ----------
    params           : tune_lgbm()에서 반환된 best_params
    scale_pos_weight : 클래스 가중치

    Returns
    -------
    model : 학습된 LGBMClassifier
    """
    print("\n########## LightGBM 학습 (Early Stopping) ##########")
    t0 = time.time()

    model = lgb.LGBMClassifier(
        **params,
        n_estimators=3000,                  # 충분히 크게 → early stopping이 결정
        objective='binary',
        metric='binary_logloss',
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=True),  # 50라운드 개선 없으면 중단
            log_evaluation(period=100),                        # 100 라운드마다 출력
        ],
    )

    print(f"\n최적 트리 수 : {model.best_iteration_}")
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

    imp_df.to_csv(os.path.join(results_dir, 'lgbm_feature_importance.csv'),
                  index=False, encoding='utf-8-sig')

    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['Feature'], top['Importance'], color='cornflowerblue')
    ax.set_xlabel('Importance (split count)')
    ax.set_title(f'LightGBM — Feature Importance Top {top_n}')
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'lgbm_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [저장] lgbm_feature_importance.csv / lgbm_feature_importance.png")
    return imp_df
