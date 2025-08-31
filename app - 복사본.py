# --- Deterministic env (must be FIRST lines) ---
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# -----------------------------------------------

import streamlit as st
import pandas as pd
from components import header # <-- header만 남기고 모두 주석 처리
from components import header, web_scraper, text_preprocessor, data_merger, eda_viewer, feature_analyzer
from components import topic_analyzer # ✨ 1. 새로 만든 모듈 임포트

# --- 1. 페이지 설정 및 세션 상태 초기화 ---
st.set_page_config(layout="wide")

# 모든 세션 상태 변수들을 앱 실행 시점에 한번만 초기화
if 'xls_object_tab3' not in st.session_state: st.session_state.xls_object_tab3 = None
if 'scraped_df' not in st.session_state: st.session_state.scraped_df = None
if 'merged_df' not in st.session_state: st.session_state.merged_df = None
if 'final_df_for_analysis' not in st.session_state: st.session_state.final_df_for_analysis = None
if 'preproc_df' not in st.session_state: st.session_state.preproc_df = None
if 'final_df' not in st.session_state: st.session_state.final_df = None
if 'count_dtm_df' not in st.session_state: st.session_state.count_dtm_df = None
if 'cos_sim_df' not in st.session_state: st.session_state.cos_sim_df = None
if 'tfidf_dtm_df' not in st.session_state: st.session_state.tfidf_dtm_df = None
if 'tfidf_cos_sim_df' not in st.session_state: st.session_state.tfidf_cos_sim_df = None
if 'highword_df' not in st.session_state: st.session_state.highword_df = None
if 'topic_df' not in st.session_state: st.session_state.topic_df = None

# --- 2. 공통 UI ---
header.show()

# --- ✨ 변경점 1: 사이드바 로직 통합 ---
with st.sidebar:
    st.header("📂 파일 선택")
    st.info("[3단계] 범용 데이터 도구에서 사용할 엑셀 파일을 업로드하세요.")
    uploaded_file_tab3 = st.file_uploader(
        "분석할 엑셀 파일", 
        type=['xlsx','xls'], 
        key="tab3_uploader",
        label_visibility="collapsed" # 라벨 숨기기
    )
    if uploaded_file_tab3:
        # 파일을 업로드하면 즉시 세션 상태에 저장
        st.session_state.xls_object_tab3 = pd.ExcelFile(uploaded_file_tab3)

# --- 3. 탭 생성 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "[1단계] 웹 스크래핑", 
    "[2단계] 텍스트 전처리", 
    "[3단계] 범용 데이터 도구", 
    "[4단계] 텍스트의 특성 및 특성 벡터",
    "[5단계] 개벽 토픽 추출"  # 탭 추가
])

# --- 탭 1, 2 (변경 없음) ---
with tab1:
    web_scraper.show()
    
with tab2:
    st.header("텍스트 전처리")
    uploaded_file_for_preproc = st.file_uploader("전처리할 엑셀 파일을 업로드하세요.", type=['xlsx', 'xls'], key="tab2_file_uploader")
    if uploaded_file_for_preproc:
        df_to_preprocess = pd.read_excel(uploaded_file_for_preproc)
        text_preprocessor.show(df_to_preprocess)
    
# --- ✨ 변경점 2: 탭 3 로직 단순화 ---
with tab3:
    st.header("범용 데이터 결합 및 탐색")
    
    # 이제 사이드바에서 업로드된 파일이 있는지 세션 상태만 확인하면 됩니다.
    if st.session_state.xls_object_tab3:
        data_merger.show(st.session_state.xls_object_tab3)
        st.divider()
        eda_viewer.show(st.session_state.final_df_for_analysis if st.session_state.final_df_for_analysis is not None else None)
    else:
        st.info("이 도구를 사용하려면 사이드바에서 엑셀 파일을 업로드해주세요.")

# --- 탭 4 (변경 없음) ---
with tab4:
    feature_analyzer.show()

# ✨ 3. 새로 추가된 탭 내용 정의
with tab5:
    topic_analyzer.show()