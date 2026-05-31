# =============================================================
# run_all.py — 3개 모델 전체 학습 & 통합 성능 비교 저장
# =============================================================
# 실행: python models_train/run_all.py
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from preprocess import load_and_preprocess

import train_decision_tree
import train_lightgbm
import train_xgboost


def main():
    print("=" * 60)
    print("  KKBox 고객이탈 예측 — 전체 모델 학습 시작")
    print("=" * 60 + "\n")

    results = []

    print("── [1/3] Decision Tree ──────────────────────────────")
    results.append(train_decision_tree.main())

    print("\n── [2/3] LightGBM ───────────────────────────────────")
    results.append(train_lightgbm.main())

    print("\n── [3/3] XGBoost ────────────────────────────────────")
    results.append(train_xgboost.main())

    # ── 통합 성능 테이블 저장 ─────────────────────────────────
    print("\n" + "=" * 60)
    results_df = pd.DataFrame(results)
    results_df.to_csv('./results/model_performance.csv', index=False)
    print("▶ 통합 성능 테이블 저장: ./results/model_performance.csv")
    print(results_df.to_string(index=False))

    # ── 통합 ROC 커브 저장 ────────────────────────────────────
    print("\n▶ 통합 ROC 커브 생성 중...")
    X_train, X_val, y_train, y_val, feature_cols = load_and_preprocess()

    models = {
        'Decision Tree': joblib.load('./models/decision_tree.pkl'),
        'LightGBM'     : joblib.load('./models/lightgbm.pkl'),
        'XGBoost'      : joblib.load('./models/xgboost.pkl'),
    }

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc = next(r['AUC'] for r in results if r['Model'] == name)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — 모델 비교')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/roc_curve.png', dpi=150)
    plt.close()
    print("  통합 ROC 커브 저장: ./results/roc_curve.png")

    # ── meta.pkl 저장 (Streamlit 시뮬레이터용) ───────────────
    meta = {
        'feature_cols': feature_cols,
        'results'     : {r['Model']: r for r in results},
    }
    joblib.dump(meta, './models/meta.pkl')
    print("  meta.pkl 저장: ./models/meta.pkl")

    print("\n✅ 전체 완료!")
    print("   모델  : ./models/")
    print("   결과  : ./results/")


if __name__ == '__main__':
    main()
