import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_tab_trend(df):
    if df is None or df.empty:
        st.warning("분석 가능한 데이터가 없습니다.")
        return

    df_tab1 = df.copy()
    
    # 1. 날짜 데이터 타입 안정성 확보
    df_tab1['멤버십종료일'] = pd.to_datetime(df_tab1['멤버십종료일'])
    df_tab1 = df_tab1.sort_values(by='멤버십종료일')

    # 2. groupby().apply() 대신 초고속 .agg() 방식으로 대치 (오류 완벽 방지)
    daily_summary = (
        df_tab1
        .groupby('멤버십종료일', as_index=False)
        .agg(
            전체_만료고객=('실제_결제금액', 'count'),
            총_리스크매출=('실제_결제금액', 'sum'),
            실제_이탈고객=('이탈여부', lambda x: (x == 1).sum()),
            # 이탈여부가 1인 데이터의 결제금액 합산
            실제_이탈매출=('실제_결제금액', lambda x: df_tab1.loc[x.index, '실제_결제금액'].where(df_tab1.loc[x.index, '이탈여부'] == 1).sum())
        )
    )

    # 3. 데이터 공백 체크 위치를 위로 이동
    if daily_summary.empty:
        st.warning("분석 가능한 데이터가 없습니다.")
        return

    # 4. 파생 지표 계산
    daily_summary['이탈률'] = (
        daily_summary['실제_이탈고객'] / daily_summary['전체_만료고객'] * 100
    ).fillna(0)

    daily_summary['예상_방어매출'] = daily_summary['실제_이탈매출'] * 0.20

    # ==================================================
    # KPI 계산
    # ==================================================
    peak_idx = daily_summary['총_리스크매출'].idxmax()
    peak_day = daily_summary.loc[peak_idx, '멤버십종료일']

    # 날짜 연산 오류 방지를 위해 형변환 확정 후 연산
    campaign_day = pd.to_datetime(peak_day) - pd.Timedelta(days=3)

    lost_idx = daily_summary['실제_이탈매출'].idxmax()

    total_lost = daily_summary['실제_이탈매출'].sum()
    total_saved = daily_summary['예상_방어매출'].sum()

    # ==================================================
    # Title
    # ==================================================
    st.markdown("<h3 style='color:#0F172A; margin-top:10px;'>📊 당월 리스크 추이 및 마케팅 골든타임 분석션</h3>", unsafe_allow_html=True)
    st.caption(
        "실제 만료 스케일과 실제 이탈 손실을 비교하여 CRM 캠페인 최적 집행 시점을 도출합니다."
    )

    # ==================================================
    # KPI Cards
    # ==================================================
    k1, k2= st.columns(2)
    k1.metric("🎯 최대 리스크일", peak_day.strftime("%Y-%m-%d"))
    k2.metric("📢 캠페인 시작일", campaign_day.strftime("%Y-%m-%d"))

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    # ==================================================
    # LEFT : MAIN CHART
    # ==================================================
    with col_left:
        fig_main = go.Figure()

        customdata = list(zip(
            daily_summary['전체_만료고객'],
            daily_summary['실제_이탈고객'],
            daily_summary['이탈률']
        ))

        # 전체 만료 리스크
        fig_main.add_trace(
            go.Scatter(
                x=daily_summary['멤버십종료일'],
                y=daily_summary['총_리스크매출'],
                mode='lines',
                name='전체 만료 리스크액',
                line=dict(width=3, color='#00E5FF'),
                fill='tozeroy',
                fillcolor='rgba(0,229,255,0.08)',
                customdata=customdata,
                hovertemplate="""
                <b>%{x|%Y-%m-%d}</b><br><br>
                전체 만료 고객 : %{customdata[0]:,.0f}명<br>
                실제 이탈 고객 : %{customdata[1]:,.0f}명<br>
                이탈률 : %{customdata[2]:.1f}%<br>
                리스크 매출 : NT$ %{y:,.0f}<br>
                <extra></extra>
                """
            )
        )

        # 실제 이탈 손실
        fig_main.add_trace(
            go.Scatter(
                x=daily_summary['멤버십종료일'],
                y=daily_summary['실제_이탈매출'],
                mode='lines+markers',
                name='❌ 실제 이탈 손실액',
                line=dict(width=3, color='#FF2A85'),
                marker=dict(size=7, color='#FF2A85'),
                fill='tozeroy',
                fillcolor='rgba(255,42,133,0.12)',
                hovertemplate="""
                <b>%{x|%Y-%m-%d}</b><br>
                실제 이탈 손실 : NT$ %{y:,.0f}
                <extra></extra>
                """
            )
        )
        
        fig_main.add_vrect(
            # .timestamp() * 1000 을 통해 숫자로 변환
            x0=campaign_day.timestamp() * 1000,
            x1=peak_day.timestamp() * 1000,
            fillcolor="rgba(0,229,255,0.12)",
            line_width=0,
            annotation_text="🎯 CRM 골든타임",
            annotation_position="top left"
        )

        # ==================================================
        # 광고 시작일
        # ==================================================
        fig_main.add_vline(
            x=campaign_day.timestamp() * 1000,  # 숫자로 변환
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="D-3 광고 시작",
            annotation_position="top"
        )

        # ==================================================
        # 최대 실제 이탈일
        # ==================================================
        lost_day = daily_summary.loc[lost_idx, '멤버십종료일']
        
        fig_main.add_annotation(
            x=lost_day.timestamp() * 1000,  # 숫자로 변환
            y=daily_summary.loc[lost_idx, '실제_이탈매출'],
            text="🚨 최대 이탈",
            showarrow=True,
            arrowhead=2,
            bgcolor="white"
        )

        fig_main.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                title="멤버십 만료 예정일",
                type="date",
                showgrid=True,
                gridcolor="#E2E8F0",
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                title="매출 스케일 (NT$)",
                showgrid=True,
                gridcolor="#E2E8F0"
            )
        )

        st.plotly_chart(fig_main, use_container_width=True)

    # ==================================================
    # RIGHT PANEL
    # ==================================================
    with col_right:
        # 1. 게이지 차트 정의 및 스타일 조정
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=total_saved,
                title={
                    'text': "🎯 예상 방어 가능 매출",
                    'font': {'color': '#1E293B', 'size': 14, 'family': 'Noto Sans KR'}
                },
                number={
                    'valueformat': ',.0f',
                    'prefix': 'NT$ ',
                    'font': {'color': '#1E293B', 'size': 22, 'family': 'Noto Sans KR'}
                },
                gauge={
                    'axis': {
                        'range': [None, max(total_lost, 1)],
                        'tickfont': {'color': '#64748B', 'size': 10}
                    },
                    'bar': {'color': "#3B82F6"}, # 단정한 슬레이트 블루로 리브랜딩
                    'bgcolor': "#F1F5F9",
                    'steps': [
                        {'range': [0, max(total_lost, 1)], 'color': 'rgba(71, 85, 105, 0.04)'}
                    ]
                }
            )
        )

        fig_gauge.update_layout(
            height=220, # 줄어든 여백으로 컴팩트하게 배치
            margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # 1. f-string과 HTML 구조가 충돌하지 않도록 변수를 먼저 문자열로 포맷팅합니다.
        # 1. 변수 포맷팅 선행
        peak_day_str = peak_day.strftime('%Y-%m-%d')
        campaign_day_str = campaign_day.strftime('%Y-%m-%d')
        total_saved_int = int(total_saved)

        # 2. 들여쓰기 없이 한 줄로 압축한 HTML 문자열 구성
        html_content = (
            "<div style='background:#F8FAFC; padding:20px; border-radius:8px; border:1px solid #E2E8F0; color:#334155; font-size:13.5px; line-height:1.6; font-family:sans-serif;'>"
            f"<h4 style='margin-top:0 !important; color:#1E293B; font-size:15px; font-weight:600; margin-bottom:12px;'>💡 데이터 기반 CRM 전략 가이드</h4>"
            "<div style='margin-bottom:16px;'>"
            "<span style='background:#E2E8F0; color:#475569; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:600; margin-right:4px;'>집중 타겟</span>"
            "<span style='font-weight:500; color:#1E293B;'>수동 결제 및 1년 미만 유저</span>"
            "</div>"
            "<ul style='padding-left:18px; margin:0; color:#475569;'>"
            f"<li style='margin-bottom:12px;'><strong style='color:#1E293B;'>마케팅 골든타임 도출</strong><br>당월 리스크 피크일은 <span style='color:#3B82F6; font-weight:600;'>{peak_day_str}</span>입니다. 효율 극대화를 위해 캠페인은 3일 전인 <span style='color:#3B82F6; font-weight:600;'>{campaign_day_str}</span>부터 집중 집행해야 합니다.</li>"
            f"<li style='margin-bottom:12px;'><strong style='color:#1E293B;'>예상 방어 매출 효과</strong><br>고위험군 유저를 대상으로 적정 오퍼를 제공해 20%만 방어하더라도 <span style='color:#1E293B; font-weight:600;'>NT$ {total_saved_int:,}</span> 규모의 매출 소실을 즉시 보호할 수 있습니다.</li>"
            "<li><strong style='color:#1E293B;'>비자발적 이탈 예방 행동 요령</strong><br>월말 만료 리스크 규모가 평소 대비 <b>5배 이상</b> 크게 뛰어오르는 패턴을 보입니다. 결제 실패로 인한 이탈을 줄이기 위해 골든타임 시작일에 <b>결제 수단 사전 점검 캠페인</b>을 연동하는 것을 권장합니다.</li>"
            "</ul>"
            "</div>"
        )

        # 3. 마크다운 대신 공식 HTML 컴포넌트로 주입
        st.html(html_content)