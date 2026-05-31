# =============================================================
# Page 1. 📊 데이터 탐색 (EDA)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

st.set_page_config(page_title="데이터 탐색 (EDA)", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    features     = pd.read_csv('./data/train_features.csv')
    transactions = pd.read_csv('./data/transactions_v2.csv')
    train_msnos  = set(features['msno'])
    transactions = transactions[transactions['msno'].isin(train_msnos)]
    return features, transactions


features, transactions = load_data()

st.title("📊 데이터 탐색 (EDA)")


# =============================================================
# 1-1. 이탈률 전체 현황
# =============================================================

st.subheader("전체 이탈률 현황")

total    = len(features)
churned  = int(features['is_churn'].sum())
retained = total - churned

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 고객 수", f"{total:,}명")
col2.metric("이탈 고객",    f"{churned:,}명")
col3.metric("유지 고객",    f"{retained:,}명")
col4.metric("이탈률",       f"{churned/total*100:.2f}%")

col_pie, col_desc = st.columns([1, 1])

with col_pie:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [retained, churned],
        labels=['유지', '이탈'],
        autopct='%1.1f%%',
        colors=['#4C9BE8', '#E85C5C'],
        startangle=90,
    )
    ax.set_title('이탈 vs 유지')
    st.pyplot(fig)
    plt.close()

with col_desc:
    st.markdown("""
    **데이터 개요**
    - 출처: KKBox Music Streaming (Kaggle)
    - 기준일: 2017년 3월 1일
    - 이탈 정의: 멤버십 만료 후 30일 내 미갱신
    """)
    st.dataframe(
        features['is_churn']
        .value_counts()
        .rename(index={0: '유지', 1: '이탈'})
        .rename('고객 수')
        .to_frame(),
        width='stretch',
    )

st.markdown("---")


# =============================================================
# 1-2. 카테고리별 이탈률
# =============================================================

st.subheader("카테고리별 이탈률")

tab1, tab2, tab3 = st.tabs(["💳 결제 수단", "📅 요금제 기간", "📱 등록 채널"])

with tab1:
    merged_tx = features[['msno', 'is_churn']].merge(
        transactions[['msno', 'payment_method_id']].drop_duplicates('msno'),
        on='msno', how='left'
    )
    top10 = merged_tx['payment_method_id'].value_counts().head(10).index
    churn_method = (
        merged_tx[merged_tx['payment_method_id'].isin(top10)]
        .groupby('payment_method_id')['is_churn']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('mean', ascending=False)
    )
    churn_method.columns = ['결제수단 ID', '이탈률', '고객수']
    churn_method['결제수단 ID'] = churn_method['결제수단 ID'].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(churn_method['결제수단 ID'], churn_method['이탈률'], color='#4C9BE8')
    ax.set_xlabel('결제수단 ID')
    ax.set_ylabel('이탈률')
    ax.set_title('결제수단별 이탈률 (상위 10)')
    ax.set_ylim(0, churn_method['이탈률'].max() * 1.2)
    for bar, val in zip(bars, churn_method['이탈률']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontsize=8)
    st.pyplot(fig)
    plt.close()

with tab2:
    main_plans = [30, 60, 90, 180, 360, 365]
    merged_plan = features[['msno', 'is_churn']].merge(
        transactions[['msno', 'payment_plan_days']].drop_duplicates('msno'),
        on='msno', how='left'
    )
    churn_plan = (
        merged_plan[merged_plan['payment_plan_days'].isin(main_plans)]
        .groupby('payment_plan_days')['is_churn']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('payment_plan_days')
    )
    churn_plan.columns = ['요금제(일)', '이탈률', '고객수']
    churn_plan['요금제(일)'] = churn_plan['요금제(일)'].astype(int).astype(str) + '일'

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(churn_plan['요금제(일)'], churn_plan['이탈률'], color='#E8A84C')
    ax.set_xlabel('요금제 기간')
    ax.set_ylabel('이탈률')
    ax.set_title('요금제 기간별 이탈률')
    ax.set_ylim(0, 1.15)
    for bar, val in zip(bars, churn_plan['이탈률']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)
    plt.close()
    st.caption("※ 30일 요금제가 가장 많고 이탈률이 낮음. 60일 이상 장기 요금제는 이탈률 급증 (데이터 불균형 주의)")

with tab3:
    via_label = {3: '앱(3)', 4: '웹(4)', 7: '모바일(7)', 9: '기타(9)', 13: 'API(13)'}
    churn_via = (
        features.groupby('registered_via')['is_churn']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('mean', ascending=False)
    )
    churn_via['registered_via'] = (
        churn_via['registered_via']
        .map(via_label)
        .fillna(churn_via['registered_via'].astype(str))
    )
    churn_via.columns = ['등록 채널', '이탈률', '고객수']

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(churn_via['등록 채널'], churn_via['이탈률'], color='#5CE89A')
    ax.set_xlabel('등록 채널')
    ax.set_ylabel('이탈률')
    ax.set_title('등록 채널별 이탈률')
    ax.set_ylim(0, churn_via['이탈률'].max() * 1.2)
    for bar, val in zip(bars, churn_via['이탈률']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)
    plt.close()

st.markdown("---")


# =============================================================
# 1-3. 청취 시간 분포
# =============================================================

st.subheader("청취 시간 분포")

secs = features['total_secs'].copy()
secs = secs[(secs > 0) & (secs < secs.quantile(0.99))]
secs_hours = secs / 3600

col_hist, col_stat = st.columns([2, 1])

with col_hist:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(secs_hours, bins=60, color='#4C9BE8', edgecolor='white', alpha=0.8)
    ax.set_xlabel('총 청취 시간 (시간)')
    ax.set_ylabel('고객 수')
    ax.set_title('고객별 총 청취 시간 분포 (상위 1% 제외)')
    st.pyplot(fig)
    plt.close()

with col_stat:
    st.markdown("**청취 시간 통계 (시간)**")
    st.dataframe(
        secs_hours.describe().rename({
            'count': '고객 수', 'mean': '평균', 'std': '표준편차',
            'min': '최솟값', '25%': '1사분위', '50%': '중앙값',
            '75%': '3사분위', 'max': '최댓값',
        }).round(1).to_frame(name='값'),
        width='stretch',
    )

st.markdown("**청취 완료율 & 스킵률 분포**")

col_c, col_s = st.columns(2)

with col_c:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(features['completion_rate'].dropna(), bins=50,
            color='#5CE89A', edgecolor='white', alpha=0.8)
    ax.set_xlabel('완료율')
    ax.set_ylabel('고객 수')
    ax.set_title('청취 완료율 분포')
    st.pyplot(fig)
    plt.close()

with col_s:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(features['skip_rate'].dropna(), bins=50,
            color='#E85C5C', edgecolor='white', alpha=0.8)
    ax.set_xlabel('스킵률')
    ax.set_ylabel('고객 수')
    ax.set_title('스킵률 분포')
    st.pyplot(fig)
    plt.close()