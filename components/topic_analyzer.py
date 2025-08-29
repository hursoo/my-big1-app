# components/topic_analyzer.py

import streamlit as st
import pandas as pd
import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt
# from kneed import KneeLocator  <- 이 부분은 이제 필요 없습니다.
import warnings
import ast

# 한글 폰트 설정
try:
    import koreanize_matplotlib
except ImportError:
    st.error("koreanize_matplotlib 라이브러리가 필요합니다. `pip install koreanize_matplotlib` 명령어로 설치해주세요.")

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 계산 함수들 ---

def calculate_log_likelihoods(corpus, progress_bar, status_text, train_steps=500, seed=100):
    """Log-likelihood 계산 (Streamlit UI 업데이트 기능 추가)"""
    ntopics = []
    log_likelihoods = []
    total_steps = 20
    
    for k in range(1, total_steps + 1):
        progress = k / total_steps
        progress_bar.progress(progress)
        status_text.text(f"Log-likelihood 계산 중... (k={k}/{total_steps})")

        mdl = tp.LDAModel(k=k, seed=seed)
        for doc in corpus:
            mdl.add_doc(doc)
        mdl.burn_in = 50
        mdl.train(train_steps, workers=1)
        ntopics.append(k)
        log_likelihoods.append(mdl.ll_per_word)
    
    status_text.text("Log-likelihood 계산 완료!")
    return ntopics, log_likelihoods

def calculate_coherence(corpus, progress_bar, status_text, k_start=2, k_end=20, train_steps=500, seed=100):
    """Coherence 점수 계산 (Streamlit UI 업데이트 기능 추가)"""
    ntopics = []
    coherence_scores = []
    total_steps = k_end - k_start + 1
    
    for i, k in enumerate(range(k_start, k_end + 1)):
        progress = (i + 1) / total_steps
        progress_bar.progress(progress)
        status_text.text(f"Coherence 점수 계산 중... (k={k}/{k_end})")

        mdl = tp.LDAModel(k=k, seed=seed, tw=tp.TermWeight.PMI)
        for doc in corpus:
            mdl.add_doc(doc)
        mdl.burn_in = 50
        mdl.train(train_steps, workers=1)
        
        coh = tp.coherence.Coherence(mdl, coherence='c_v')
        coherence_value = coh.get_score()
        ntopics.append(k)
        coherence_scores.append(coherence_value)

    status_text.text("Coherence 점수 계산 완료!")
    return ntopics, coherence_scores

def safe_str_to_list(s):
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return str(s).split()


# --- 메인 UI 함수 ---

def show():
    st.header("[5단계] 개벽 토픽 추출 💫")
    
    st.info("토픽 모델링(LDA)을 통해 문서 집합의 주요 주제를 추출합니다. 최적의 토픽 개수(k)를 찾기 위한 두 가지 지표를 계산하고 시각화합니다.")

    uploaded_file = st.file_uploader(
        "분석할 엑셀 파일을 업로드하세요. (전처리 완료된 'doc_split_12gram' 컬럼 필요)", 
        type=['xlsx', 'xls']
    )

    if uploaded_file:
        try:
            gb_df = pd.read_excel(uploaded_file)
            if 'doc_split_12gram' not in gb_df.columns:
                st.error("'doc_split_12gram' 컬럼을 찾을 수 없습니다. 파일 형식을 확인해주세요.")
                return
            
            gb_df['doc_split_12gram'] = gb_df['doc_split_12gram'].apply(safe_str_to_list)
            corpus = gb_df['doc_split_12gram'].to_list()
            st.success(f"파일 로딩 완료! 총 {len(corpus)}개의 문서가 준비되었습니다.")
            
            # --- 최적 토픽 수 탐색 섹션 ---
            st.subheader("1. 최적 토픽 수(k) 탐색")
            st.markdown("""
            모델의 성능을 평가하는 두 가지 지표를 계산하여 최적의 토픽 수를 결정합니다.
            - **모델 혼잡도 (Log-likelihood):** 값이 가장 낮은 지점 (Minimum point)이 후보가 됩니다.
            - **토픽 일관성 (Coherence):** 값이 가장 높은 지점 (Peak point)이 가장 좋은 후보입니다.
            """)

            col1, col2 = st.columns(2)

            # --- 모델 혼잡도 (Log-likelihood) 개별 실행 ---
            with col1:
                st.markdown("#### 모델 혼잡도 (Log-likelihood)")
                if st.button("혼잡도 계산 시작"):
                    with st.spinner('Log-likelihood 계산 중...'):
                        ll_progress = st.progress(0)
                        ll_status = st.empty()
                        ntopics_ll, log_likelihoods = calculate_log_likelihoods(corpus, ll_progress, ll_status)
                        
                        st.session_state.log_likelihood_results = pd.DataFrame({
                            'k': ntopics_ll,
                            'Log-likelihood': log_likelihoods
                        }).set_index('k')

            if 'log_likelihood_results' in st.session_state:
                with col1:
                    st.write("##### 계산 결과")
                    st.dataframe(st.session_state.log_likelihood_results)

                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(st.session_state.log_likelihood_results.index, st.session_state.log_likelihood_results['Log-likelihood'], marker='o', linestyle='--')
                    ax1.set_xlabel("Number of topics (k)")
                    ax1.set_ylabel("Log-likelihood")
                    ax1.set_title("Log-likelihood per Number of Topics")
                    ax1.set_xticks(st.session_state.log_likelihood_results.index)
                    ax1.grid(True, alpha=0.5)
                    st.pyplot(fig1)
                    
                    # --- ✨ 변경점: 가장 낮은 지점(Minimum Point) 계산 및 표시 ---
                    min_point_k = st.session_state.log_likelihood_results['Log-likelihood'].idxmin()
                    st.success(f"혼잡도 기반 추천 k (최저점): **{min_point_k}**")


            # --- 토픽 일관성 (Coherence) 개별 실행 ---
            with col2:
                st.markdown("#### 토픽 일관성 (Coherence Score)")
                if st.button("일관성 점수 계산 시작"):
                    with st.spinner('Coherence 점수 계산 중...'):
                        coh_progress = st.progress(0)
                        coh_status = st.empty()
                        ntopics_coh, coherence_scores = calculate_coherence(corpus, coh_progress, coh_status)

                        st.session_state.coherence_results = pd.DataFrame({
                            'k': ntopics_coh,
                            'Coherence (c_v)': coherence_scores
                        }).set_index('k')
            
            if 'coherence_results' in st.session_state:
                with col2:
                    st.write("##### 계산 결과")
                    st.dataframe(st.session_state.coherence_results)

                    coh_results_df = st.session_state.coherence_results
                    best_k = coh_results_df['Coherence (c_v)'].idxmax()
                    
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    ax3.plot(coh_results_df.index, coh_results_df['Coherence (c_v)'], marker='o', linestyle='--', color='mediumseagreen', label='Topic Coherence (c_v)')
                    ax3.axvline(x=best_k, color='tomato', linestyle=':', linewidth=2, label=f'Best k = {best_k}')
                    ax3.set_xlabel("Number of Topics(k)")
                    ax3.set_ylabel("Coherence Score (c_v)")
                    ax3.set_title("Coherence Score by Number of Topics")
                    ax3.set_xticks(coh_results_df.index)
                    ax3.legend()
                    st.pyplot(fig3)
                    st.success(f"일관성 기반 추천 k (Peak): **{best_k}**")

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            st.warning("엑셀 파일의 'doc_split_12gram' 컬럼 내용이 `['단어1', '단어2']` 와 같은 리스트 형태로 되어있는지 확인해주세요.")