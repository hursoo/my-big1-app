import streamlit as st
import pandas as pd
import io # df.info() 출력을 위해 필요

def show(df: pd.DataFrame):
    """
    데이터프레임을 받아 기본적인 탐색적 데이터 분석 결과를 보여주는 함수.
    (상관관계 분석 제외)
    """
    st.header("🔎 탐색적 데이터 분석 (EDA)")

    if df is None or df.empty:
        st.warning("분석할 데이터가 없습니다.")
        return

    # st.expander를 사용하여 섹션별로 내용을 접고 펼 수 있게 만듭니다.
    with st.expander("① 데이터 기본 정보"):
        st.write("- **데이터 형태 (Shape):**", f"{df.shape[0]}행, {df.shape[1]}열")
        
        # df.info()의 결과를 텍스트로 가져오기 위한 처리
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with st.expander("② 결측치 확인"):
        missing_values = df.isnull().sum()
        st.dataframe(missing_values[missing_values > 0].sort_values(ascending=False), use_container_width=True)
        if missing_values.sum() == 0:
            st.success("🎉 모든 데이터에 결측치가 없습니다!")

    with st.expander("③ 기초 통계 (숫자형 데이터)"):
        st.dataframe(df.describe().transpose(), use_container_width=True)