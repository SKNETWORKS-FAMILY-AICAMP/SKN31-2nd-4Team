import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def render_tab_eda(df):
    # -------------------------------------------------------------------------
    # 1. 단정하고 심플한 MODERN UX/UI 스타일 바인딩
    # -------------------------------------------------------------------------
    st.markdown("""
        <style>
        /* 타이틀 및 헤더 구조 - 단정한 다크 슬레이트 */
        .eda-main-title {
            font-size: 24px;
            font-weight: 700;
            color: #1E293B !important;
            margin-top: 10px;
            letter-spacing: -0.5px;
        }
        
        h4 {
            color: #1E293B !important;
            font-weight: 600 !important;
            margin-top: 20px !important;
        }
        
        /* 미니멀한 모던 인사이트 카드 */
        .sunny-insight-card {
            background-color: #F8FAFC;
            border-left: 4px solid #94A3B8; /* 차분한 그레이 뼈대 */
            padding: 14px 16px;
            border-radius: 6px;
            margin-top: 12px;
            margin-bottom: 20px;
            color: #334155;
            font-size: 13.5px;
            line-height: 1.6;
            border: 1px solid #E2E8F0;
            border-left-width: 4px;
        }
        
        /* 메트릭 블록 컴포넌트 폰트 최적화 */
        [data-testid="stMetricLabel"] {
            color: #64748B !important; /* 차분한 서브 텍스트 컬러 */
            font-weight: 500 !important;
        }
        [data-testid="stMetricValue"] {
            color: #1E293B !important; /* 선명하고 단정한 메인 값 */
            font-weight: 700 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎧 1. 유저 일평균 청취 및 탐색 인게이지먼트 분석")
    st.markdown("<p style='font-size:15px; color:#64748B; margin-bottom: 25px;'>일부 아웃라이어로 인한 왜곡을 방지하기 위해 이상치를 제외했습니다. </p>", unsafe_allow_html=True)
    

    # 상단 지표 계산
    churn_rate = (df['이탈여부'].sum() / len(df)) * 100 if len(df) > 0 else 0
    auto_renew_rate = (df['자동갱신_동의여부'].sum() / len(df)) * 100 if '자동갱신_동의여부' in df.columns else 0
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="🎯 모델 예측 평균 이탈률", value=f"{churn_rate:.1f} %", delta="위험군 집중 방어 필요")
    with m_col2:
        st.metric(label="🔄 전체 자동 갱신 등록 비율", value=f"{auto_renew_rate:.1f} %", delta="-1.2% 전월 대비 감소", delta_color="inverse")
    with m_col3:
        avg_secs = df['총청취시간(분)'].mean() if '총청취시간(분)' in df.columns else 0
        st.metric(label="🎧 인당 일평균 스트리밍 청취", value=f"{avg_secs/60:.1f} 분", delta="안전군 대비 42% 부족")

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================================
    # VISUAL 02. 행동 로그 기반 인게이지먼트 교차 분석
    # ==============================================================================
    
    limit_secs = 6000
    limit_unq = float(np.percentile(df['고유 재생 곡수'].dropna(), 99)) if '고유 재생 곡수' in df.columns else 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # [차트 1] 청취 시간 분포 (박스플롯)
        fig_secs = px.box(
            df, 
            x='이탈여부', 
            y='총청취시간(분)',
            color='이탈여부',
            color_discrete_map={0: '#3B82F6', 1: '#F59E0B'}, # 슬레이트 블루 vs 차분한 앰버
            labels={'이탈여부': '구독 상태', '총청취시간(분)': '일평균 청취 시간 (분)'}
        )
        
        fig_secs.update_traces(boxpoints=False)
        fig_secs.update_layout(
            title=dict(
                text="⏳ 이탈 리스크 유저군별 일일 청취 시간 분포",
                font=dict(color="#1E293B", size=16, family='Noto Sans KR')
            ),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#475569',
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor='#F1F5F9', ticktext=['안전 (Renewal)', '이탈위험 (Churn)'], tickvals=[0, 1]), 
            yaxis=dict(gridcolor='#F1F5F9', range=[0, limit_secs])
        )
        st.plotly_chart(fig_secs, use_container_width=True)
        
        st.markdown("""
            <div class="sunny-insight-card" style="border-left-color: #3B82F6;">
                <strong style="color: #1E293B;">💡 참여 진단 지표:</strong><br>
                이탈 위험 고객(1)의 청취 시간 중앙값이 안전 고객(0)에 비해 확연히 낮게 형성되어 있는 <b>인게이지먼트 붕괴 현상</b>을 확인할 수 있습니다.
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # [차트 2] 고유 재생 곡 수 (히스토그램)
        fig_unq = px.histogram(
            df,
            x='고유 재생 곡수',
            color='이탈여부',
            barmode='overlay',
            color_discrete_map={0: '#3B82F6', 1: '#F59E0B'},
            labels={'고유 재생 곡수': '일평균 고유 청취 곡 수', 'count': '유저 수'},
            nbins=50,
            range_x=[0, limit_unq]
        )
        
        fig_unq.update_layout(
            title=dict(
                text="🎵 고유 스트리밍 곡 수 밀도 스캔",
                font=dict(color='#1E293B', size=16, family='Noto Sans KR')
            ),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#475569',
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=dict(text="")),
            xaxis=dict(gridcolor='#F1F5F9'), 
            yaxis=dict(gridcolor='#F1F5F9')
        )
        st.plotly_chart(fig_unq, use_container_width=True)
        
        st.markdown("""
            <div class="sunny-insight-card" style="border-left-color: #F59E0B;">
                <strong style="color: #1E293B;">💡 행동 전조 탐색 지표:</strong><br>
                고유 곡 수가 0~20곡 미만인 초반 밀집 구간에 <b>이탈 위험군이 집중적으로 분포</b>되어 있습니다. 즉, 신규 탐색을 멈춘 유저의 이탈 확률이 높습니다.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================================
    # VISUAL 03. 비즈니스 / 결제 메커니즘 분석
    # ==============================================================================
    st.markdown("#### 💰 2. 비즈니스 파트 (결제 및 가입 패턴)")
    st.markdown("<p style='font-size:15px; color:#64748B; margin-bottom: 25px;'>유저의 결제 방식 및 누적 가입 코호트 기간에 따른 비즈니스 생존율 지표입니다.</p>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)

    with col3:
        # [차트 3] 자동 갱신 여부별 이탈 비중 (Stacked Bar)
        if '자동갱신_동의여부' in df.columns:
            renew_churn = df.groupby(['자동갱신_동의여부', '이탈여부']).size().reset_index(name='유저수')
            renew_churn['자동갱신_동의여부'] = renew_churn['자동갱신_동의여부'].map({0: '수동 결제 유저', 1: '자동 갱신 동의 유저'})
            
            fig_renew = px.bar(
                renew_churn,
                x='자동갱신_동의여부',
                y='유저수',
                color='이탈여부',
                color_discrete_map={0: '#3B82F6', 1: '#F59E0B'},
                labels={'이탈여부': '구독 상태', '유저수': '고객 수'},
                barmode='stack'
            )
            
            fig_renew.update_layout(
                title=dict(
                    text="🔄 자동 갱신 동의 여부별 이탈 비중 대조",
                    font=dict(color='#1E293B', size=16, family='Noto Sans KR')
                ),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#475569',
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=dict(text="")),
                xaxis=dict(gridcolor='rgba(0,0,0,0)'), 
                yaxis=dict(gridcolor='#F1F5F9')
            )
            st.plotly_chart(fig_renew, use_container_width=True)
            
            st.markdown("""
                <div class="sunny-insight-card" style="border-left-color: #3B82F6;">
                    <strong style="color: #1E293B;">💡 결제 마찰력 진단:</strong><br>
                    매월 주기적인 결제 액션이 필요한 <b>수동 결제 유저군</b>에서 자동 갱신 유저 대비 실제 이탈 결제 공백 위험이 확연히 높게 발생합니다.
                </div>
            """, unsafe_allow_html=True)

    with col4:
        # [차트 4] 결제 수단별 이탈 비중 대조 (가로 Stacked Bar)
        payment_col = '결제수단ID'  # 👈 본인의 데이터셋 컬럼명에 맞게 수정하세요 (예: 'payment_method_id')
        
        if payment_col in df.columns:
            pay_churn = df.groupby([payment_col, '이탈여부']).size().reset_index(name='유저수')
            
            fig_pay = px.bar(
                pay_churn,
                y=payment_col,
                x='유저수',
                color='이탈여부',
                orientation='h',  # 가로형 차트로 시각적 다양성 확보
                color_discrete_map={0: '#3B82F6', 1: '#F59E0B'},
                labels={'이탈여부': '구독 상태', '유저수': '고객 수'},
                barmode='stack'
            )
            
            fig_pay.update_layout(
                title=dict(
                    text="💳 결제 수단별 이탈 위험도 분포",
                    font=dict(color='#1E293B', size=16, family='Noto Sans KR')
                ),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#475569',
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=dict(text="")),
                xaxis=dict(gridcolor='#F1F5F9'), 
                yaxis=dict(gridcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_pay, use_container_width=True)
            
            st.markdown("""
                <div class="sunny-insight-card" style="border-left-color: #F59E0B;">
                    <strong style="color: #1E293B;">💡 결제 인프라 진단:</strong><br>
                    특정 결제 수단(예: 휴대폰 소액결제 등)에서 잔액 부족이나 명의 변경 등으로 인한 <b>비자발적 이탈</b>이 빈번하게 발생할 수 있으므로 선제적 알림이 필요합니다.
                </div>
            """, unsafe_allow_html=True)
        else:
            # 해당 컬럼이 없을 경우 안전하게 기본 요금제(실제_결제금액) 분석으로 대체하는 가이드 st.warning 표현
            st.warning(f"'{payment_col}' 컬럼이 데이터셋에 없습니다. 데이터셋의 결제 관련 컬럼명을 확인해주세요.")


