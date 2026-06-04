import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='color:#00AFEC; margin-bottom:0;'>🎵 KKBOX Portal</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; font-size:11px; margin-top:0;'>PREDICTIVE ACTION CONTROL</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### 🎛️ 시뮬레이션 제어 패널")
        
        risk_threshold = st.slider(
            "🚨 이탈 위험 판정 임계치 (Threshold)", 
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            help="이 확률 이상인 유저를 고위험 타겟군으로 분류합니다."
        )
        
        conversion_rate = st.slider(
            "🎯 CRM 마케팅 예상 전환율 (%)", 
            min_value=0.0, max_value=100.0, value=15.0, step=0.5,
            help="프로모션을 받고 마음을 돌려 잔류할 유저의 비율입니다."
        )
        
        m_cost = st.number_input(
            "💰 유저당 마케팅 발송 비용 (NT$)", 
            min_value=0, max_value=100, value=10, step=1,
            help="알림톡, 푸시, 쿠폰 비용 등 인당 소요되는 마케팅 비용입니다."
        )
        
    return {
        "risk_threshold": risk_threshold,
        "conversion_rate": conversion_rate,
        "m_cost": m_cost
    }