# components/topic_analyzer.py

import streamlit as st
import pandas as pd
import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import warnings
import ast  # ✨ 1. ast 라이브러리를 임포트합니다.

# 한글 폰트 설정
try:
    import koreanize_matplotlib
except ImportError:
    st.error("koreanize_matplotlib 라이브러리가 필요합니다. `pip install koreanize_matplotlib` 명령어로 설치해주세요.")

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 계산 함수들 (사용자 코드 기반) ---

def calculate_log_likelihoods(corpus, progress_bar, status_text, train_steps=500, seed=100):
    """Log-likelihood 계산 (Streamlit UI 업데이트 기능 추가)"""
    ntopics = []
    log_likelihoods = []
    total_steps = 20
    
    for k in range(1, total_steps + 1):
        # UI 업데이트
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
        # UI 업데이트
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

# ✨ 2. 어떤 데이터 형식이든 안전하게 리스트로 변환하는 함수를 추가합니다.
def safe_str_to_list(s):
    try:
        # 문자열이 "['a', 'b']" 형태일 경우, 실제 리스트 ['a', 'b']로 변환
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        # 변환에 실패하면 (예: "a b c" 형태의 일반 텍스트), 공백으로 나눔
        return str(s).split()


# --- 메인 UI 함수 ---

def show():
    st.header("[5단계] 개벽 토픽 추출 💫")
    
    st.info("토픽 모델링(LDA)을 통해 문서 집합의 주요 주제를 추출합니다. 최적의 토픽 개수(k)를 찾기 위한 두 가지 지표를 계산하고 시각화합니다.")

    # 1. 파일 업로드
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
            
            # str 형식의 리스트를 실제 리스트로 변환
            gb_df['doc_split_12gram'] = gb_df['doc_split_12gram'].apply(safe_str_to_list)
            corpus = gb_df['doc_split_12gram'].to_list()
            st.success(f"파일 로딩 완료! 총 {len(corpus)}개의 문서가 준비되었습니다.")

            # 2. 최적 토픽 수 탐색
            st.subheader("1. 최적 토픽 수(k) 탐색")
            st.markdown("""
            모델의 성능을 평가하는 두 가지 지표를 계산하여 최적의 토픽 수를 결정합니다.
            - **모델 혼잡도 (Log-likelihood):** 값이 급격히 꺾이는 지점 (Elbow point)이 후보가 됩니다.
            - **토픽 일관성 (Coherence):** 값이 가장 높은 지점 (Peak point)이 가장 좋은 후보입니다.
            """)

            if st.button("최적 토픽 수 계산 시작 (시간이 다소 소요됩니다)"):
                
                col1, col2 = st.columns(2)

                # --- 모델 혼잡도 (Log-likelihood) ---
                with col1:
                    st.markdown("#### 모델 혼잡도 (Log-likelihood)")
                    with st.spinner('Log-likelihood 계산 중...'):
                        ll_progress = st.progress(0)
                        ll_status = st.empty()
                        ntopics_ll, log_likelihoods = calculate_log_likelihoods(corpus, ll_progress, ll_status)

                    # 원본 그래프
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(ntopics_ll, log_likelihoods, marker='o', linestyle='--')
                    ax1.set_xlabel("Number of topics (k)")
                    ax1.set_ylabel("Log-likelihood")
                    ax1.set_title("Log-likelihood per Number of Topics")
                    ax1.set_xticks(ntopics_ll)
                    ax1.grid(True, alpha=0.5)
                    st.pyplot(fig1)

                    # 변화율 그래프
                    loglikelihood_diff = np.diff(np.array(log_likelihoods))
                    kl = KneeLocator(x=ntopics_ll[1:], y=loglikelihood_diff, curve='concave', direction='increasing')
                    elbow_point = kl.elbow
                    st.success(f"혼잡도 기반 추천 k (Elbow): **{elbow_point}**")

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.plot(ntopics_ll[1:], loglikelihood_diff, marker='o', label="Change in Log-likelihood")
                    if elbow_point:
                        ax2.axvline(x=elbow_point, color='red', linestyle='--', label=f"Elbow (k={elbow_point})")
                    ax2.set_xlabel('Number of Topics (k)')
                    ax2.set_ylabel('Change in Log-likelihood')
                    ax2.set_title('Change in Log-likelihood by Number of Topics')
                    ax2.grid(alpha=0.5)
                    ax2.legend()
                    st.pyplot(fig2)

                # --- 토픽 일관성 (Coherence) ---
                with col2:
                    st.markdown("#### 토픽 일관성 (Coherence Score)")
                    with st.spinner('Coherence 점수 계산 중...'):
                        coh_progress = st.progress(0)
                        coh_status = st.empty()
                        ntopics_coh, coherence_scores = calculate_coherence(corpus, coh_progress, coh_status)

                    best_k_index = np.argmax(coherence_scores)
                    best_k = ntopics_coh[best_k_index]
                    st.success(f"일관성 기반 추천 k (Peak): **{best_k}**")

                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    ax3.plot(ntopics_coh, coherence_scores, marker='o', linestyle='--', color='mediumseagreen', label='Topic Coherence (c_v)')
                    ax3.axvline(x=best_k, color='tomato', linestyle=':', linewidth=2, label=f'Best k = {best_k}')
                    ax3.set_xlabel("Number of Topics(k)")
                    ax3.set_ylabel("Coherence Score (c_v)")
                    ax3.set_title("Coherence Score by Number of Topics")
                    ax3.set_xticks(ntopics_coh)
                    ax3.legend()
                    st.pyplot(fig3)

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            st.warning("엑셀 파일의 'doc_split_12gram' 컬럼 내용이 `['단어1', '단어2']` 와 같은 리스트 형태로 되어있는지 확인해주세요.")