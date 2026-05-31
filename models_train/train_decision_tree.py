# =============================================================
# train_decision_tree.py — Decision Tree 학습 & 평가
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, roc_auc_score, f1_score, roc_curve

from preprocess import load_and_preprocess


def main():
    # ── 전처리 ───────────────────────────────────────────────
    X_train, X_val, y_train, y_val, feature_cols = load_and_preprocess()

    # ── 모델 정의 ────────────────────────────────────────────
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
    )

    # ── 학습 ─────────────────────────────────────────────────
    print("▶ Decision Tree 학습 중...")
    model.fit(X_train, y_train)
    print("  학습 완료!\n")

    # ── 평가 ─────────────────────────────────────────────────
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    ll  = log_loss(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    f1  = f1_score(y_val, y_pred)

    print("▶ 평가 결과 (Decision Tree)")
    print(f"  Log Loss : {ll:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  F1       : {f1:.4f}\n")

    # ── ROC 커브 저장 ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'Decision Tree (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Decision Tree')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/roc_decision_tree.png', dpi=150)
    plt.close()
    print("  ROC 커브 저장: ./results/roc_decision_tree.png")

    # ── 피처 중요도 저장 ─────────────────────────────────────
    importance_df = pd.DataFrame({
        'feature'    : feature_cols,
        'importance' : model.feature_importances_,
    }).sort_values('importance', ascending=False).head(10)

    importance_df.to_csv('./results/importance_decision_tree.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'][::-1], importance_df['importance'][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance — Decision Tree')
    plt.tight_layout()
    plt.savefig('./results/importance_decision_tree.png', dpi=150)
    plt.close()
    print("  피처 중요도 저장: ./results/importance_decision_tree.png\n")

    # ── 모델 저장 ─────────────────────────────────────────────
    joblib.dump(model, './models/decision_tree.pkl')
    print("▶ 모델 저장 완료: ./models/decision_tree.pkl")

    # ── 성능 결과 반환 (통합 비교용) ─────────────────────────
    return {'Model': 'Decision Tree', 'Log Loss': round(ll, 4), 'AUC': round(auc, 4), 'F1': round(f1, 4)}


if __name__ == '__main__':
    main()
