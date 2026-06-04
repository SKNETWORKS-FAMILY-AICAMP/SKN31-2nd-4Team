import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ──────────────────────────────────────────────
# Decision Tree 하이퍼파라미터 튜닝 근거
#
# max_depth        : 트리 깊이 제한 → 과적합 방지 핵심 파라미터
#                    너무 깊으면 train에 과적합, 너무 얕으면 underfitting
# min_samples_leaf : 리프 노드 최소 샘플 수 → 작은 값은 노이즈에 과적합
# min_samples_split: 분기 최소 샘플 수 → 희귀 패턴 과적합 방지
# class_weight     : 'balanced' 고정 → 이탈율 9.46% 클래스 불균형 대응
#                    소수 클래스(이탈)에 자동으로 높은 가중치 부여
# ──────────────────────────────────────────────

PARAM_GRID = {
    'max_depth'        : [5, 7, 10, 15, 20],
    'min_samples_leaf' : [5, 10, 50, 100],
    'min_samples_split': [10, 20, 50, 100],
}


def tune_dt(X_train, y_train, cv: int = 5) -> dict:
    """
    GridSearchCV로 Decision Tree 최적 하이퍼파라미터 탐색

    Parameters
    ----------
    X_train, y_train : 학습 데이터
    cv               : StratifiedKFold 폴드 수

    Returns
    -------
    best_params : dict
    """
    print("########## Decision Tree GridSearchCV 시작 ##########")
    t0 = time.time()

    base_model = DecisionTreeClassifier(
        class_weight='balanced',  # 클래스 불균형 대응
        random_state=42
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring='roc_auc',        # 불균형 데이터 → AUC 기준
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\n최적 파라미터 : {grid.best_params_}")
    print(f"최적 AUC(CV) : {grid.best_score_:.4f}")
    print(f"소요 시간    : {time.time() - t0:.1f}s")

    return grid.best_params_


def train_dt(X_train, y_train, params: dict = None):
    """
    최적 파라미터로 Decision Tree 학습

    Parameters
    ----------
    params : tune_dt()에서 반환된 best_params (None이면 기본값 사용)

    Returns
    -------
    model : 학습된 DecisionTreeClassifier
    """
    print("\n########## Decision Tree 학습 ##########")
    t0 = time.time()

    if params is None:
        params = {'max_depth': 10, 'min_samples_leaf': 50, 'min_samples_split': 50}

    model = DecisionTreeClassifier(
        **params,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    print(f"학습 완료 ({time.time() - t0:.1f}s)")
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

    # CSV 저장
    imp_df.to_csv(os.path.join(results_dir, 'dt_feature_importance.csv'),
                  index=False, encoding='utf-8-sig')

    # PNG 저장
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['Feature'], top['Importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Decision Tree — Feature Importance Top {top_n}')
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'dt_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [저장] dt_feature_importance.csv / dt_feature_importance.png")
    return imp_df
