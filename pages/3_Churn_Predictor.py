# =============================================================
# Page 3. 🔮 이탈 예측 시뮬레이터
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="이탈 예측 시뮬레이터", page_icon="🔮", layout="wide")

st.title("🔮 이탈 예측 시뮬레이터")
st.markdown("가상의 고객 정보를 입력하면 이탈 확률을 예측합니다.")


# =============================================================
# 모델 로딩
# =============================================================

@st.cache_resource
def load_models():
    meta = joblib.load('./models/meta.pkl')
    lgbm = joblib.load('./models/lightgbm.pkl')
    dt   = joblib.load('./models/decision_tree.pkl')
    xgb  = joblib.load('./models/xgboost.pkl')
    return meta, lgbm, dt, xgb


@st.cache_data
def load_features():
    return pd.read_csv('./data/train_features.csv')


try:
    meta, lgbm, dt, xgb = load_models()
except FileNotFoundError:
    st.error("모델 파일이 없습니다. model_train.py를 먼저 실행해주세요.")
    st.stop()

feature_cols = meta['feature_cols']


# =============================================================
# 3-1. 입력 폼
# =============================================================

st.subheader("고객 정보 입력")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**👤 회원 정보**")
    tenure_days    = st.slider("가입 후 경과일",   0, 3000, 365)
    bd             = st.slider("나이",             10, 80, 30)
    gender_enc     = st.selectbox("성별",
                        options=[0, 1, -1],
                        format_func=lambda x: {0: '남성', 1: '여성', -1: '미입력'}[x])
    city           = st.slider("도시 코드",         1, 21, 1)
    registered_via = st.selectbox("등록 채널",
                        options=[3, 4, 7, 9, 13],
                        format_func=lambda x: {3:'앱', 4:'웹', 7:'모바일', 9:'기타', 13:'API'}[x])

with col_b:
    st.markdown("**💳 결제 정보**")
    tx_count         = st.slider("결제 횟수",              1, 50, 5)
    avg_paid         = st.slider("평균 결제금액 (원)",      0, 500, 149)
    auto_renew_ratio = st.slider("자동갱신 비율",           0.0, 1.0, 0.8)
    cancel_ratio     = st.slider("취소 비율",               0.0, 1.0, 0.05)
    plan_days_mean   = st.slider("평균 요금제 기간 (일)",   1, 365, 30)
    days_until_expire= st.slider("만료까지 남은 일수",     -30, 365, 15)
    days_since_last_tx=st.slider("마지막 결제 후 경과일",   0, 365, 10)
    discount_ratio   = st.slider("할인 의존도",             0.0, 1.0, 0.1)
    avg_discount_rate= st.slider("평균 할인율",             0.0, 1.0, 0.05)
    plan_variety     = st.slider("요금제 다양성",           1, 10, 1)
    payment_methods  = st.slider("결제수단 수",             1, 5, 1)
    downgrade_count  = st.slider("다운그레이드 횟수",       0, 10, 0)

with col_c:
    st.markdown("**🎵 청취 정보**")
    log_count         = st.slider("총 접속 일수",           1, 790, 200)
    num_unq           = st.slider("총 청취 곡 수",          0, 100000, 5000)
    total_secs        = st.slider("총 청취 시간 (초)",      0, 10000000, 1000000)
    
    active_days = st.slider( "최근 활동 일수", 0, 365, 30)
    days_since_last_listen = st.slider("마지막 청취 후 경과일",0, 365, 1)
    plays_recent30 = st.slider("최근 30일 재생 수",0, 100000,1000)
    plays_prev30 = st.slider("이전 30일 재생 수",0, 100000,800)

    completion_rate   = st.slider("완료율",                 0.0, 1.0, 0.5)
    skip_rate         = st.slider("스킵률",                 0.0, 1.0, 0.2)
    listen_regularity = st.slider("청취 규칙성 (일당 곡수)", 0.0, 100.0, 25.0)

    listen_trend_ratio = plays_recent30 / max(plays_prev30, 1)

# engagement_score 자동 계산
engagement_score = (
    completion_rate * 0.4 +
    (1 - skip_rate) * 0.3 +
    (log_count / max(plan_days_mean, 1)) * 0.3
)

st.markdown("---")


# =============================================================
# 3-2. 예측 실행
# =============================================================

input_dict = {
    'tenure_days'       : tenure_days,
    'bd'                : bd,
    'gender_enc'        : gender_enc,
    'city'              : city,
    'registered_via'    : registered_via,
    'tx_count'          : tx_count,
    'avg_paid'          : avg_paid,
    'avg_discount_rate' : avg_discount_rate,
    'discount_ratio'    : discount_ratio,
    'auto_renew_ratio'  : auto_renew_ratio,
    'cancel_ratio'      : cancel_ratio,
    'plan_days_mean'    : plan_days_mean,
    'plan_variety'      : plan_variety,
    'payment_methods'   : payment_methods,
    'days_since_last_tx': days_since_last_tx,
    'days_until_expire' : days_until_expire,
    'downgrade_count'   : downgrade_count,
    'total_secs'        : total_secs,
    'num_unq'           : num_unq,
    'log_count'         : log_count,
    'completion_rate'   : completion_rate,
    'skip_rate'         : skip_rate,
    'listen_regularity' : listen_regularity,
    
    'active_days'           : active_days,
    'days_since_last_listen': days_since_last_listen,
    'plays_recent30'        : plays_recent30,
    'plays_prev30'          : plays_prev30,
    'listen_trend_ratio'    : listen_trend_ratio,

    'engagement_score'  : engagement_score,
}

X_input = pd.DataFrame([input_dict])[feature_cols]

if st.button("🔮 이탈 확률 예측", width='stretch'):

    prob_lgbm = lgbm.predict_proba(X_input)[0][1]
    prob_dt   = dt.predict_proba(X_input)[0][1]
    prob_xgb  = xgb.predict_proba(X_input)[0][1]
    prob_avg  = (prob_lgbm + prob_dt + prob_xgb) / 3

    st.subheader("예측 결과")

    if prob_avg >= 0.5:
        risk_label, risk_color = "🔴 고위험", "#E85C5C"
    elif prob_avg >= 0.3:
        risk_label, risk_color = "🟡 중위험", "#E8A84C"
    else:
        risk_label, risk_color = "🟢 저위험", "#5CE89A"

    st.markdown(
        f"<h2 style='text-align:center; color:{risk_color};'>"
        f"{risk_label} — 이탈확률 {prob_avg*100:.1f}%</h2>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LightGBM",      f"{prob_lgbm*100:.1f}%")
    col2.metric("Decision Tree", f"{prob_dt*100:.1f}%")
    col3.metric("XGBoost",       f"{prob_xgb*100:.1f}%")
    col4.metric("앙상블 평균",   f"{prob_avg*100:.1f}%")

    # 게이지 바
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([''], [1],        color='#e0e0e0', height=0.5)
    ax.barh([''], [prob_avg], color=risk_color, height=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel('이탈 확률')
    ax.set_title(f'이탈 확률 게이지: {prob_avg*100:.1f}%')
    ax.text(min(prob_avg + 0.02, 0.95), 0,
            f'{prob_avg*100:.1f}%', va='center', fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

    with st.expander("입력 피처 요약 보기"):
        summary_df = pd.DataFrame(input_dict.items(), columns=['피처', '입력값'])
        st.dataframe(summary_df, width='stretch', hide_index=True)

st.markdown("---")


# =============================================================
# 3-3. 고위험 고객군 분석
# =============================================================

st.subheader("고위험 고객군 분석")
st.markdown("학습 데이터 기준으로 이탈 고객과 유지 고객의 피처 분포를 비교합니다.")

try:
    features_df = load_features()
    churned  = features_df[features_df['is_churn'] == 1]
    retained = features_df[features_df['is_churn'] == 0]

    risk_cols = [
        ('auto_renew_ratio',  '자동갱신 비율'),
        ('cancel_ratio',      '취소 비율'),
        ('days_until_expire', '만료 남은 일수'),
        ('completion_rate',   '청취 완료율'),
        ('skip_rate',         '스킵률'),
        ('engagement_score',  '인게이지먼트 점수'),
    ]

    col_r1, col_r2 = st.columns(2)

    for i, (col, label) in enumerate(risk_cols):
        if col not in features_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(retained[col].dropna(), bins=40, alpha=0.6,
                label='유지', color='#4C9BE8', density=True)
        ax.hist(churned[col].dropna(),  bins=40, alpha=0.6,
                label='이탈', color='#E85C5C', density=True)
        ax.set_title(label)
        ax.set_xlabel(col)
        ax.set_ylabel('밀도')
        ax.legend()

        if i % 2 == 0:
            col_r1.pyplot(fig)
        else:
            col_r2.pyplot(fig)
        plt.close()

except FileNotFoundError:
    st.warning("train_features.csv 파일이 없습니다. feature_engineering.py를 먼저 실행해주세요.")