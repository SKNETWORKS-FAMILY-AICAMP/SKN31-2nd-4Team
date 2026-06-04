# app.py
import streamlit as st
from utils.data_loader import load_data
from utils.sidebar import render_sidebar
from utils.headers import render_header

# 탭 별 화면 임포트
from views.tab_eda import render_tab_eda
from views.tab_solution import render_tab_solution
from views.tab_trend import render_tab_trend
from views.tab_model import render_tab_model

# 1. 기본 페이지 설정 및 스타일 주입
st.set_page_config(
    page_title="KKBOX Pulse Center",
    page_icon="🎵",
    layout="wide"
)

with open("utils/style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

data = load_data()

# 2. 직관적인 변수명으로 언패킹(Unpacking)
df = data["user"]
df_shap = data["shap"]  # 🚨 이것도 명시적으로 꺼내두세요!
df_feat = data["feat"]
df_perf = data["perf"]

# 3. 공통 컴포넌트 호출
# 사이드바에서 설정한 파라미터가 필요한 경우 crm_params["risk_threshold"] 형태로 꺼내 쓸 수 있습니다.
# crm_params = render_sidebar() 
render_header(df)

st.markdown("<br>", unsafe_allow_html=True)

# 4. 탭 시스템 인터랙션 분기 제어
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 전체 가입자 EDA 리포트", 
    "⚙️ LightGBM 검증 성능 리포트",
    "📈 리스크 추이 및 골든타임",
    "🎯 위험 등급별 고객 CRM"
])

with tab1:
    render_tab_eda(df)

with tab2:
    render_tab_model(df_perf, df_feat, df_shap, df)
    
with tab3:
    render_tab_trend(df)

with tab4:
    render_tab_solution(df)