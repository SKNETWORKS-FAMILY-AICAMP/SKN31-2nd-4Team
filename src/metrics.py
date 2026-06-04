import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss,
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

# ──────────────────────────────────────────────
# 평가 지표 설명
#
# AUC-ROC  : 클래스 불균형에 강함. 이탈/유지 구분 능력 측정 (1에 가까울수록 좋음)
# F1-Score : Precision과 Recall의 조화평균. 이탈(소수 클래스) 예측 성능에 민감
# Log Loss : 예측 확률의 정확도 측정. 낮을수록 확률 보정이 잘 됨
# ──────────────────────────────────────────────


def evaluate_model(model_name: str, y_true, y_pred, y_proba) -> dict:
    """
    단일 모델 평가 지표 계산

    Parameters
    ----------
    model_name : str
    y_true     : 실제 레이블
    y_pred     : 예측 레이블 (0/1)
    y_proba    : 이탈 확률 (0~1)

    Returns
    -------
    dict : 모델명 + 주요 지표
    """
    return {
        'Model'    : model_name,
        'AUC'      : round(roc_auc_score(y_true, y_proba), 6),
        'F1'       : round(f1_score(y_true, y_pred, zero_division=0), 6),
        'Log Loss' : round(log_loss(y_true, y_proba), 6),
        'Accuracy' : round(accuracy_score(y_true, y_pred), 6),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 6),
        'Recall'   : round(recall_score(y_true, y_pred, zero_division=0), 6),
    }


def save_classification_report(model_name: str, y_true, y_pred, results_dir: str = './results'):
    """
    Classification report를 CSV로 저장
    """
    os.makedirs(results_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    fname = f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
    df.to_csv(os.path.join(results_dir, fname), encoding='utf-8-sig')
    print(f"  [저장] {fname}")


def save_confusion_matrix(model_name: str, y_true, y_pred, results_dir: str = './results'):
    """
    혼동행렬 PNG로 저장
    """
    os.makedirs(results_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['유지(0)', '이탈(1)'])
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title(f'{model_name} — Confusion Matrix')
    plt.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(os.path.join(results_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [저장] {fname}")


def save_roc_curve(model_name: str, y_true, y_proba, results_dir: str = './results'):
    """
    ROC 커브 PNG로 저장 (모델별 개별 저장)

    Parameters
    ----------
    model_name : str
    y_true     : 실제 레이블
    y_proba    : 이탈 확률 (0~1)
    """
    os.makedirs(results_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()

    fname = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    fig.savefig(os.path.join(results_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [저장] {fname}")


def save_roc_curve_combined(records: list, results_dir: str = './results'):
    """
    세 모델 ROC 커브를 한 그래프에 겹쳐서 저장 (비교용)

    Parameters
    ----------
    records : list of dict
        {'name': 모델명, 'y_true': ..., 'y_proba': ...} 형태의 리스트
    """
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ['steelblue', 'darkorange', 'green']
    for i, rec in enumerate(records):
        fpr, tpr, _ = roc_curve(rec['y_true'], rec['y_proba'])
        auc = roc_auc_score(rec['y_true'], rec['y_proba'])
        ax.plot(fpr, tpr, linewidth=2, color=colors[i],
                label=f"{rec['name']} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — 모델 비교')
    ax.legend(loc='lower right')
    ax.grid(True)
    plt.tight_layout()

    fig.savefig(os.path.join(results_dir, 'roc_curve_combined.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [저장] roc_curve_combined.png")


def save_model_comparison(records: list, results_dir: str = './results'):
    """
    모델 비교표를 CSV로 저장 및 출력

    Parameters
    ----------
    records : list of dict
        evaluate_model() 결과를 담은 리스트
    """
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(records).sort_values('AUC', ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print("           📊 모델 성능 비교")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    print(f"  [저장] model_comparison.csv")
    return df
