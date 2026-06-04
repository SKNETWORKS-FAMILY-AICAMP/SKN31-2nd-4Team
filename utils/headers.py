# components/header.py
import streamlit as st

def render_header(df):
    # ==============================================================================
    # 마스터 마트 기반 실시간 비즈니스 ROI 계산 로직
    # ==============================================================================
    total_expiration_users = len(df)
    total_risk_revenue_nt = int(df['실제_결제금액'].sum()) if total_expiration_users > 0 else 0
    avg_subscription_fee_nt = int(df['실제_결제금액'].mean()) if total_expiration_users > 0 else 0

    # [수정 하이라이트] 하드코딩 대신 머신러닝 모델이 탐지한 실제 이탈 고위험군(1) 모수 추정 연동
    if 'predicted_churn' in df.columns:
        model_detected_churn_users = len(df[df['predicted_churn'] == 1])
    else:
        # 혹시 모를 에러 방지용 백업 라인 (실제 Churn의 20%로 추정치 계산)
        model_detected_churn_users = int(len(df[df['이탈여부'] == 1]) * 0.20) if '이탈여부' in df.columns else 0
        
    # 그로스 마케팅 타겟 오퍼를 통한 CRM 방어 성공률을 실무 표준 스펙(20%)으로 대입하여 가치 환산
    expected_saved_users = int(model_detected_churn_users * 0.20) 
    expected_recovery_value_nt = int(expected_saved_users * avg_subscription_fee_nt)

    # 대만 달러(NT$) -> 한화(KRW) 환산 가치 (1 TWD = 약 42원 기준 병기)
    total_risk_revenue_krw = int(total_risk_revenue_nt * 42)
    avg_subscription_fee_krw = int(avg_subscription_fee_nt * 42)
    expected_recovery_value_krw = int(expected_recovery_value_nt * 42)

    # ==============================================================================
    # HTML 배너 및 상단 카드 4개 렌더링
    # ==============================================================================
    st.markdown("""
        <div class="growth-header">
            <h2>📈 KKBOX 구독자 이탈 예측 및 잔류(CRM) 전략 시뮬레이션</h2>
            <p>SK Networks Family AI 캠프 31기 | 2nd 프로젝트 4팀</p>
        </div>
    """, unsafe_allow_html=True)

    # 4개 영역 밸런스 배치
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
            <div class="ux-metric-card">
                <div class="ux-card-title">🔴 당월 멤버십 만료 예정자</div>
                <div class="ux-card-value">{total_expiration_users:,} 명</div>
                <div class="ux-card-sub">결제 주기 30일 플랜 기준</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
            <div class="ux-metric-card">
                <div class="ux-card-title">📉 Gross ARR 매출 리스크</div>
                <div class="ux-card-value" style="color: #FF2A85;">NT$ {total_risk_revenue_nt:,}</div>
                <div class="ux-card-sub">한화 약 {total_risk_revenue_krw:,}원</div>
            </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
            <div class="ux-metric-card">
                <div class="ux-card-title">💵 만료 예정자 평균 구독료</div>
                <div class="ux-card-value-blue">NT$ {avg_subscription_fee_nt:,}</div>
                <div class="ux-card-sub">한화 약 {avg_subscription_fee_krw:,}원</div>
            </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        st.markdown(f"""
            <div class="ux-metric-card" style="border-left: 5px solid #00E676;">
                <div class="ux-card-title">🎯 CRM 예상 방어 매출액</div>
                <div class="ux-card-value" style="color: #00E676;">NT$ {expected_recovery_value_nt:,}</div>
                <div class="ux-card-sub">한화 약 {expected_recovery_value_krw:,}원</div>
            </div>
        """, unsafe_allow_html=True)