# views/tab_solution.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_tab_solution(df):
    st.markdown("<h3 style='color:#0F172A; margin-top:10px;'>🎯 이탈 위험군별 처방 솔루션</h3>", unsafe_allow_html=True)
    st.caption("모델이 계산한 이탈 확률 스코어를 바탕으로 위험군별 모수를 필터링하고 타겟 CRM 액션 리스트를 도출합니다.")
    st.markdown("---")

    merged_df = df.copy()
    
    col_metrics, col_pie = st.columns([2, 3])
    
    with col_metrics:
        st.markdown("<h4 style='font-size:16px; color:#1E293B; margin-bottom:15px;'>📊 3대 리스크 세그먼트 규모</h4>", unsafe_allow_html=True)
        
        group_counts = merged_df['risk_group'].value_counts()
        high_cnt = group_counts.get('High Risk', 0)
        med_cnt = group_counts.get('Mid Risk', 0)
        low_cnt = group_counts.get('Low Risk', 0)
        total_cnt = len(merged_df) if len(merged_df) > 0 else 1
        
        st.markdown(f"""
            <div style="background-color: #FFF5F5; padding: 15px; border-radius: 8px; border-left: 5px solid #EF4444; margin-bottom: 10px;">
                <span style="font-size: 13px; color: #E53E3E; font-weight: bold;">🔴 고위험군 (High Risk)</span>
                <div style="font-size: 22px; font-weight: bold; color: #C53030;">{high_cnt:,} 명 <span style="font-size:14px; font-weight:normal; color:#E53E3E;">({high_cnt/total_cnt*100:.1f}%)</span></div>
            </div>
            <div style="background-color: #FFFBEB; padding: 15px; border-radius: 8px; border-left: 5px solid #F59E0B; margin-bottom: 10px;">
                <span style="font-size: 13px; color: #D97706; font-weight: bold;">🟡 잠재위험군 (Mid Risk)</span>
                <div style="font-size: 22px; font-weight: bold; color: #B45309;">{med_cnt:,} 명 <span style="font-size:14px; font-weight:normal; color:#D97706;">({med_cnt/total_cnt*100:.1f}%)</span></div>
            </div>
            <div style="background-color: #F0FDF4; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981;">
                <span style="font-size: 13px; color: #059669; font-weight: bold;">🟢 안전군 (Low Risk)</span>
                <div style="font-size: 22px; font-weight: bold; color: #047857;">{low_cnt:,} 명 <span style="font-size:14px; font-weight:normal; color:#059669;">({low_cnt/total_cnt*100:.1f}%)</span></div>
            </div>
        """, unsafe_allow_html=True)

    with col_pie:
        # 3분류 카테고리를 직관적으로 비교하는 도넛 차트
        fig_pie = px.pie(
            merged_df, 
            names='risk_group', 
            hole=0.4,
            color='risk_group',
            color_discrete_map={'High Risk': '#EF4444', 'Mid Risk': '#F59E0B', 'Low Risk': '#10B981'},
            title="🎯 전체 만료 대상자 위험군 비율"
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10), height=250,
            legend=dict(orientation="v", y=0.5, x=0.9)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================================
    # VISUAL 02. 인터랙티브 필터 컨트롤러 및 마케팅 명부 추출
    # ==============================================================================
    st.markdown("#### 🎛️ 리스크 등급 선택 및 타겟 대처 리스트")
    
    # 상단 라디오 버튼 필터 인터랙션 인터페이스
    selected_group = st.radio(
        "조회할 리스크 등급 세그먼트를 선택하세요:",
        options=['전체 보기', 'High Risk', 'Mid Risk', 'Low Risk'],
        horizontal=True
    )
    
    # 필터링 분기
    if selected_group != '전체 보기':
        filtered_df = merged_df[merged_df['risk_group'] == selected_group].copy()
    else:
        filtered_df = merged_df.copy()

    # 위험 수준별 개인화 CRM 비즈니스 전략 가공 함수
    def assign_crm_action(group):
        if group == 'High Risk': 
            return "🔴 [긴급] 30일 무료 이용권 + 앱 복귀 리타겟팅 푸시 오퍼"
        elif group == 'Mid Risk': 
            return "🟡 [경고] 월간 자동 갱신 유도 40% 스페셜 쿠폰 발송"
        else: 
            return "🔵 [유지] 개인화 플레이리스트 뉴스레터 자동 큐레이션 발송"
            
    filtered_df['추천 CRM 액션 전략'] = filtered_df['risk_group'].apply(assign_crm_action)
    
    # 확률 상위 순으로 정렬 및 대시보드 렌더링 가독성을 위한 상위 500개 슬라이싱
    display_df = filtered_df.sort_values(by='churn_probability', ascending=False).head(500)
    
    # 최종 매핑 상태 컬럼 지정 (초 단위 청취시간 대신 앞서 전처리한 분 단위 컬럼 결합)
    table_cols = ['고객_ID', 'risk_group', 'churn_probability', '총청취시간(분)', '고유 재생 곡수', '추천 CRM 액션 전략']
    rename_cols = {
        '고객_ID': '고객_ID',
        'risk_group': '위험 등급',
        'churn_probability': '이탈 확률 스코어',
        '총청취시간(분)': '총청취시간(분)',
        '고유 재생 곡수': '고유 재생 곡 수'
    }
    
    # 완성형 테이블 출력
    st.dataframe(
        display_df[table_cols].rename(columns=rename_cols),
        use_container_width=True,
        hide_index=True
    )
    
    # Braze/Salesforce 마케팅 연동용 파일 다운로드 버튼 빌드 (utf-8-sig로 한글 깨짐 방지)
    csv_data = filtered_df[['고객_ID', 'risk_group', 'churn_probability', '추천 CRM 액션 전략']].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label=f"📥 선택된 세그먼트 [{selected_group}] 오디언스 명부 다운로드",
        data=csv_data,
        file_name=f"kkbox_target_{selected_group.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )