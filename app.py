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
import io

# 기존 컴포넌트들을 모두 임포트합니다.
from components import header, web_scraper, text_preprocessor, data_merger, eda_viewer, feature_analyzer, topic_analyzer

# --- 1. 페이지 설정 및 세션 상태 초기화 ---
st.set_page_config(layout="wide", page_title="빅데이터 한국근대사 분석")

# 모든 세션 상태 변수들을 앱 실행 시점에 한번만 초기화
if 'main_nav' not in st.session_state: st.session_state.main_nav = "0. 강의 소개"
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

# --- 2. 사이드바 (메인 네비게이션) ---
with st.sidebar:
    st.title("📜 분석 목차")
    
    main_menu_options = [
        "[0] 강의 소개",
        "[1] 연구질문과 자료가공",
        "[2] 특성 벡터와 토픽 추출",
        "[3] 2기 필자 그룹 간 차이 파악",
        "[4] 분석 과정 정리와 활용"
    ]
    st.session_state.main_nav = st.radio(
        "메뉴를 선택하세요:",
        main_menu_options,
        label_visibility="collapsed"
    )

# --- 3. 공통 헤더 UI ---
header.show()
st.divider()

# --- 4. 메인 화면 (사이드바 선택에 따라 동적으로 변경) ---

# [메뉴 0] 강의 소개
if st.session_state.main_nav == "[0] 강의 소개":
    tab_intro, tab_schedule, tab_qna = st.tabs(["강의 개요", "주차별 강의", "자료 및 Q&A"])
    with tab_intro:  
        st.markdown(
        """
        <br>  
        **과 목 명:** 역사학 논문쓰기 1(25-2) (M3502.008000. 강좌001). 1학점. <br>
        **담당 교수:** <a href="https://humanities.snu.ac.kr/academics/faculty?deptidx=9&md=view&profidx=116" target="_blank">허 수(許 洙)</a> (crctaper@snu.ac.kr) <br>
        **강의 시간:** 월, 14:00~14:50 <br>
        **강의 장소:** 21동(약학관) 102호 <br>  
        """,unsafe_allow_html=True)

        st.markdown("""---""")

        st.markdown(
        """
        **강의 목표:**
        - 토픽 모델링으로 한국 근대 언론자료를 분석하여 전통적인 역사 연구로 해결하지 못한 문제를 다루어본다.   
          (1920년대『개벽』잡지의 사회주의를 관찰 - 논조 변화 양상과 그 범위를 중심으로)   
        - “**연구질문과 자료 가공 → 특성 벡터와 토픽 추출 → 2기 필자 그룹간 차이 파악 → 분석 과정 정리와 활용**”의 주요 단계를, 교수자가 제공하는 정제 데이터와 디지털 분석 도구 등을 사용하여 진행한다.
        - 이 과정에 참여한 수강생들은 빅데이터를 활용한 디지털 역사분석의 기본 개념과 구체적인 과정을 습득하게 되어 자신의 분야에서 필요한 데이터 분석에 응용이 가능해진다.   
        """,unsafe_allow_html=True)

        st.markdown("""---""")

        st.markdown(
        """
        **평가 방법:**
        - 등급제 여부: **A~F** or **S/U**
        - 성적부여 방식: 절대평가
        - 주요 수업방식: 플립러닝, 이론
        - 구분: 출석(50%), 과제(50%)
        - 수업일수의 1/3을 초과하여 결석하면 성적은 "F" 또는 "U"가 됨(학칙 85조)   
          (담당교수가 불가피한 결석으로 인정하는 경우는 예외로 할 수 있음)
        
        """, unsafe_allow_html=True) # ✨ 이 옵션을 추가하면 해결됩니다.

    with tab_schedule:
        #st.subheader("🗓️ 주차별 강의")
        st.markdown("""
        ■ 본 수업은 ‘대면’과 ‘원격’을 섞은 ‘혼합원격수업’으로 진행한다.   
        ■ 대면 수업은 5회(01, 04, 08, 11, 15) 실시한다. (아래 표의 **초록색 회차/일자** 참조)   
        ■ 대면 수업시 수강생들은 노트북을 지참한다.  
        ■ 그 이외에는 원격 수업을 실시하며 플립러닝 방식으로 진행한다.   
        ■ 과제는 본 수업에서 진행하는 대상과 분석 방법을 습득하고 응용하는 수준에서 제시될 것이다.   
        ■ 수업 복습과 과제 수행시 수업에 배정된 ‘튜터’의 도움을 받을 수 있다.  
        """)
        csv_data = """대주제,회차,일자,주제,코드 파일 / 플랫폼
"1부. 연구질문과 자료 가공",1,09/01,"『개벽』의 사료적 가치와 연구 질문",
,2,09/08,텍스트 분석을 위한 환경 구축,
,3,09/15,토픽 모델링과 개벽 말뭉치,
,4,09/22,파이썬을 활용한 데이터 전처리,"gb_011_scraping(big1).ipynb
gb_021_preprocess(big1).ipynb
gb_031_make_2gram(big1).ipynb"
"2부. 특성 벡터와 토픽 추출",5,09/29,‘특성’ 및 ‘특성 벡터’ 이해하기,gb_041_feature_dtm.ipynb
,6,10/06,『개벽』 논조 파악을 위한 특성 벡터 추출,"gb_051_compare_vectorizer.ipynb
gb_061_gb_feature_dtm.ipynb"
,7,10/13,토픽모델링의 원리와 주요 모델,
,8,10/20,『개벽』의 토픽 추출 과정,
,9,10/27,파이썬을 활용한 토픽 추출의 실제 과정,"gb_071_topic_extract_basic(big1).ipynb
gb_081_topic_extract(big1).ipynb"
"3부. 2기 필자 그룹간 차이 파악",10,11/03,텍스트 속의 토픽,"gb_091_topic_in_text(big1).ipynb"
,11,11/10,시기구분과 논조 변화,"gb_101_time_period(big1).ipynb
gb_111_topic_network(big1).ipynb"
,12,11/17,2기 필자 그룹 간 차이 여부,gb_121_writer_difference(big1).ipynb
"4부. 분석 과정 정리와 활용",13,11/24,필자별 주제 차이 검토,gb_131_writer_topic_networks(big1).ipynb
,14,12/01,개벽 논조 분석 과정 정리,
,15,12/08,분석 활용을 위한 모색,streamlit web app
"""
        
        # --- ✅ [수정 완료] 모든 표 관련 코드가 with tab_schedule: 블록 안으로 들어왔습니다. ---
        df = pd.read_csv(io.StringIO(csv_data))
        df['대주제'] = df['대주제'].fillna(method='ffill')
        df['코드 파일 / 플랫폼'] = df['코드 파일 / 플랫폼'].fillna('')

        def generate_html_table(dataframe):
            # ✨ 1. 중요한 회차 목록을 여기에 정의합니다. (수정하기 쉬운 부분)
            important_sessions = [1, 4, 8, 11, 15]

            html = """
            <style>
                .styled-table {
                    border-collapse: collapse; margin: 25px 0; font-size: 0.9em;
                    font-family: sans-serif; width: 100%;
                }
                .styled-table th, .styled-table td {
                    padding: 12px 15px; border: 1px solid #dddddd;
                }
                /* ✨ 2. 헤더(첫째 줄)에 회색 음영 적용 */
                .styled-table thead tr {
                    background-color: #f2f2f2; 
                    text-align: center;
                }
                .left-align { text-align: left; }
                .center-align { text-align: center; }
                .multi-line { white-space: pre-wrap; }
            </style>
            <table class='styled-table'>
                <thead>
                    <tr>
                        <th>대주제</th><th>회차</th><th>일자</th><th>주제</th><th>코드 파일 / 플랫폼</th>
                    </tr>
                </thead>
                <tbody>
            """
            for a_topic, group in dataframe.groupby('대주제', sort=False):
                is_first_row = True
                for _, row in group.iterrows():
                    # ✨ 3. 중요한 회차인지 확인하여 스타일을 결정합니다.
                    cell_style = ""
                    if row['회차'] in important_sessions:
                        cell_style = "style='background-color: #e6ffcc;'" # 연두색

                    html += f"<tr>"
                    if is_first_row:
                        html += f"<td class='center-align' rowspan='{len(group)}'>{a_topic}</td>"
                        is_first_row = False
                    
                    # ✨ 4. 결정된 스타일을 '회차'와 '일자' 셀에 적용합니다.
                    html += f"<td class='center-align' {cell_style}>{row['회차']}</td>"
                    html += f"<td class='center-align' {cell_style}>{row['일자']}</td>"
                    html += f"<td class='left-align'>{row['주제']}</td>"
                    code_files = str(row['코드 파일 / 플랫폼']).replace('\n', '<br>')
                    html += f"<td class='left-align multi-line'>{code_files}</td>"
                    html += "</tr>"
            html += "</tbody></table>"
            return html

        st.markdown(generate_html_table(df), unsafe_allow_html=True)


    with tab_qna:
        #st.subheader("📚 자료 및 Q&A")
        #st.info("이곳에 강의 자료 및 Q&A 내용이 들어갈 예정입니다.")
        st.markdown("""
        - <a href="https://github.com/hursoo/big_k-modern_1/tree/main">교수자 깃헙 주소: big_k-modern_1</a>
        - <a href="https://github.com/hursoo/big_k-modern_1/blob/main/data/gb_data_2.1.xlsx">개벽 주요논설 코퍼스: gb_data_2.1.xlsx</a>
        - <a href="https://github.com/hursoo/big_k-modern_1/blob/main/data/%EA%B7%BC%ED%98%84%EB%8C%80%EC%9E%A1%EC%A7%80%EC%9E%90%EB%A3%8C_20250315172708.txt">개벽 전체 기사 정보: 근현대잡지자료_20250315172708.txt</a>
                    
        """, unsafe_allow_html=True)

        st.markdown("""---""")

        st.markdown("""
        **깃허브 데이터를 일괄 다운로드**
        - 1단계: GitHub에서 폴더 주소 복사하기 -> https://github.com/hursoo/big_k-modern_1
        - 2단계: DownGit 사이트에서 붙여넣고 다운로드하기
            - 아래 링크를 클릭하여 DownGit 웹사이트로 이동합니다.
            - https://downgit.github.io/
            - 웹사이트 중앙에 있는 입력창에 방금 복사한 GitHub 폴더 주소를 붙여넣습니다. (Ctrl+V)
            - Download 버튼을 클릭합니다.
        - 3단계: 로컬 다운로드 받은 파일 압축 풀기
        - 4단계: 구글 드라이브의 big_km_history01 폴더에 업로드
            - 'data' 내 파일은 구글 드라이브의 'data' 폴더에 넣음.
            - ipynb 파일은 big_km_history01 폴더에 넣음.
        """)

        st.markdown("""---""")

        st.markdown("""
        - <a href="https://forms.gle/wHNzVuQGJ3mXbLrP8">수강 전 설문조사</a>
        """, unsafe_allow_html=True)

        

# [메뉴 1] 연구질문과 자료가공
elif st.session_state.main_nav == "[1] 연구질문과 자료가공":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "[1단계] 웹 스크래핑", 
        "[2단계] 텍스트 전처리", 
        "[3단계] 범용 데이터 도구", 
        "[4단계] 텍스트의 특성",
        "[5단계] 개벽 토픽 추출"
    ])
    with tab1:
        web_scraper.show()
    with tab2:
        st.header("텍스트 전처리")
        uploaded_file_for_preproc = st.file_uploader("전처리할 엑셀 파일을 업로드하세요 (**5개 기사로 된 샘플 파일: ron10_data.xlsx**)", type=['xlsx', 'xls'], key="tab2_file_uploader")
        if uploaded_file_for_preproc:
            df_to_preprocess = pd.read_excel(uploaded_file_for_preproc)
            text_preprocessor.show(df_to_preprocess)
    with tab3:
        st.header("범용 데이터 결합 및 탐색")
        uploaded_file_tab3 = st.file_uploader("분석할 엑셀 파일", type=['xlsx','xls'], key="tab3_uploader")
        if uploaded_file_tab3:
            xls_object = pd.ExcelFile(uploaded_file_tab3)
            data_merger.show(xls_object)
            st.divider()
            if 'final_df_for_analysis' in st.session_state and st.session_state.final_df_for_analysis is not None:
                 eda_viewer.show(st.session_state.final_df_for_analysis)
        else:
            st.info("이 도구를 사용하려면 위에서 엑셀 파일을 업로드해주세요.")
    with tab4:
        st.subheader("✨ [변경] 이 탭의 내용은 '특성 벡터와 토픽 추출' 메뉴로 이동했습니다.")
        st.info("좌측 사이드바에서 '2. 특성 벡터와 토픽 추출' 메뉴를 선택해주세요.")
    with tab5:
        st.subheader("✨ [변경] 이 탭의 내용은 '특성 벡터와 토픽 추출' 메뉴로 이동했습니다.")
        st.info("좌측 사이드바에서 '2. 특성 벡터와 토픽 추출' 메뉴를 선택해주세요.")

# [메뉴 2] 특성 벡터와 토픽 추출
elif st.session_state.main_nav == "[2] 특성 벡터와 토픽 추출":
    tab_feature, tab_topic = st.tabs(["특성 벡터 생성", "토픽 모델링 및 시각화"])
    with tab_feature:
        feature_analyzer.show()
    with tab_topic:
        topic_analyzer.show()

# [메뉴 3] 2기 필자 그룹 간 차이 파악
elif st.session_state.main_nav == "[3] 2기 필자 그룹 간 차이 파악":
    tab_group_diff, tab_stats = st.tabs(["그룹별 토픽 비교 분석", "통계적 유의성 검증"])
    with tab_group_diff:
        st.subheader("그룹별 토픽 비교 분석")
        st.info("이곳에서는 필자 그룹별 주요 토픽 분포와 그 차이를 시각적으로 분석하는 기능이 제공될 예정입니다.")
    with tab_stats:
        st.subheader("통계적 유의성 검증")
        st.info("그룹 간의 차이가 통계적으로 유의미한지 검증하는 기능이 제공될 예정입니다.")

# [메뉴 4] 분석 과정 정리와 활용
elif st.session_state.main_nav == "[4] 분석 과정 정리와 활용":
    tab_summary, tab_report = st.tabs(["분석 결과 요약", "보고서 생성 및 활용"])
    with tab_summary:
        st.subheader("분석 결과 요약")
        st.info("지금까지의 모든 분석 과정을 요약하고, 주요 발견점을 정리하는 기능이 제공될 예정입니다.")
    with tab_report:
        st.subheader("보고서 생성 및 활용")
        st.info("분석 결과를 바탕으로 보고서를 자동 생성하거나, 결과를 다른 포맷으로 내보내는 기능이 제공될 예정입니다.")

