# =============================================================
# Page 2. 🤖 모델 성능 비교
# =============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="모델 성능", page_icon="🤖", layout="wide")

st.title("🤖 모델 성능 비교")


# =============================================================
# 2-1. 성능 테이블
# =============================================================

st.subheader("모델별 성능 지표")

try:
    perf = pd.read_csv('./results/model_performance.csv')
except FileNotFoundError:
    st.error("model_performance.csv 파일이 없습니다. model_train.py를 먼저 실행해주세요.")
    st.stop()


def highlight_best(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    styles.loc[df['Log Loss'].idxmin(), 'Log Loss'] = 'background-color: #d4edda; font-weight: bold'
    styles.loc[df['AUC'].idxmax(),      'AUC']      = 'background-color: #d4edda; font-weight: bold'
    styles.loc[df['F1'].idxmax(),       'F1']        = 'background-color: #d4edda; font-weight: bold'
    return styles


st.dataframe(
    perf.style.apply(highlight_best, axis=None).format({
        'Log Loss': '{:.4f}', 'AUC': '{:.4f}', 'F1': '{:.4f}'
    }),
    width='stretch',
    hide_index=True,
)
st.caption("✅ 초록색: 해당 지표에서 가장 우수한 모델")

st.markdown("---")


# =============================================================
# 2-2. 지표별 막대그래프
# =============================================================

st.subheader("지표별 시각화")

colors = ['#4C9BE8', '#E8A84C', '#E85C5C']

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(perf['Model'], perf['Log Loss'], color=colors)
    ax.set_title('Log Loss (낮을수록 좋음)')
    ax.set_ylim(0, perf['Log Loss'].max() * 1.3)
    for i, v in enumerate(perf['Log Loss']):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    plt.xticks(rotation=10, fontsize=8)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(perf['Model'], perf['AUC'], color=colors)
    ax.set_title('AUC (높을수록 좋음)')
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(perf['AUC']):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    plt.xticks(rotation=10, fontsize=8)
    st.pyplot(fig)
    plt.close()

with col3:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(perf['Model'], perf['F1'], color=colors)
    ax.set_title('F1 Score (높을수록 좋음)')
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(perf['F1']):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    plt.xticks(rotation=10, fontsize=8)
    st.pyplot(fig)
    plt.close()

st.markdown("---")


# =============================================================
# 2-3. ROC 커브
# =============================================================

st.subheader("ROC 커브")

try:
    st.image('./results/roc_curve.png', width='stretch')
except Exception:
    st.warning("roc_curve.png 파일이 없습니다.")

st.markdown("---")


# =============================================================
# 2-4. SHAP 피처 중요도
# =============================================================

st.subheader("SHAP 피처 중요도 Top 10 (LightGBM)")

col_shap, col_desc = st.columns([1.5, 1])

with col_shap:
    try:
        st.image('./results/shap_importance.png', width='stretch')
    except Exception:
        st.warning("shap_importance.png 파일이 없습니다.")

with col_desc:
    st.markdown("""
    **SHAP(SHapley Additive exPlanations)**

    각 피처가 예측에 얼마나 기여하는지를 수치로 표현합니다.

    - **Mean |SHAP|** 값이 클수록 이탈 예측에 중요한 피처
    - 모든 고객에 대해 평균을 내므로 전체적인 중요도 파악에 용이
    """)
    try:
        shap_df = pd.read_csv('./results/shap_importance.csv')
        shap_df.columns = ['피처', 'Mean |SHAP|']
        shap_df['Mean |SHAP|'] = shap_df['Mean |SHAP|'].round(4)
        st.dataframe(shap_df, width='stretch', hide_index=True)
    except FileNotFoundError:
        st.warning("shap_importance.csv 파일이 없습니다.")