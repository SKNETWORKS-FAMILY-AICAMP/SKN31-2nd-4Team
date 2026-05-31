# =============================================================
# train_lightgbm.py — LightGBM 학습 & 평가
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score, f1_score, roc_curve

from preprocess import load_and_preprocess


def main():
    # ── 전처리 ───────────────────────────────────────────────
    X_train, X_val, y_train, y_val, feature_cols = load_and_preprocess()

    # 클래스 불균형 보정 비율
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # ── 모델 정의 ────────────────────────────────────────────
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )

    # ── 학습 ─────────────────────────────────────────────────
    print("▶ LightGBM 학습 중...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    print("  학습 완료!\n")

    # ── 평가 ─────────────────────────────────────────────────
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    ll  = log_loss(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    f1  = f1_score(y_val, y_pred)

    print("▶ 평가 결과 (LightGBM)")
    print(f"  Log Loss : {ll:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  F1       : {f1:.4f}\n")

    # ── ROC 커브 저장 ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'LightGBM (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — LightGBM')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/roc_lightgbm.png', dpi=150)
    plt.close()
    print("  ROC 커브 저장: ./results/roc_lightgbm.png")

    # ── SHAP 피처 중요도 저장 ────────────────────────────────
    print("▶ SHAP 계산 중...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap_importance = pd.DataFrame({
        'feature'       : feature_cols,
        'mean_abs_shap' : np.abs(sv).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).head(10)

    shap_importance.to_csv('./results/shap_importance.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(shap_importance['feature'][::-1], shap_importance['mean_abs_shap'][::-1])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 10 Feature Importance (SHAP) — LightGBM')
    plt.tight_layout()
    plt.savefig('./results/shap_importance.png', dpi=150)
    plt.close()
    print("  SHAP 저장: ./results/shap_importance.csv / shap_importance.png\n")

    # ── 모델 저장 ─────────────────────────────────────────────
    joblib.dump(model, './models/lightgbm.pkl')
    print("▶ 모델 저장 완료: ./models/lightgbm.pkl")

    # ── 성능 결과 반환 (통합 비교용) ─────────────────────────
    return {'Model': 'LightGBM', 'Log Loss': round(ll, 4), 'AUC': round(auc, 4), 'F1': round(f1, 4)}


if __name__ == '__main__':
    main()
