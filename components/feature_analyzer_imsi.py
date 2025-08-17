# =====================
# BLOCK-FEATURE (replace entire components/feature_analyzer.py)
# =====================
# --- BEGIN BLOCK-FEATURE ---
# path: components/feature_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import unicodedata

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tomotopy as tp
from tomotopy import DMRModel, TermWeight, utils


def normalize_whitespace(text: str) -> str:
    """Unicode normalize + collapse whitespace (reproducible token stream)."""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.strip().split())


# --- Helpers ---
def get_dtm_gb(df, col_name, stopw, rank_n):
    tv = TfidfVectorizer(ngram_range=(1, 1), stop_words=stopw)
    dtm = tv.fit_transform(df[col_name].astype("str"))
    term_sums = np.array(dtm.sum(axis=0)).flatten()
    highword_indices = term_sums.argsort()[-rank_n:][::-1]
    highword_list = [tv.get_feature_names_out()[i] for i in highword_indices]
    feature_dtm = dtm[:, highword_indices]
    return pd.DataFrame(feature_dtm.toarray(), columns=highword_list, index=df.index)


def set_global_seeds(seed_value=1000):
    random.seed(seed_value)
    np.random.seed(seed_value)


def transform_a_data_to_metadata(misc: dict):
    return {"metadata": str(misc["a_data"])}


# --- Core: pure training function (no Streamlit calls) ---
def run_dmr_model(
    gridL,
    lineL,
    *,
    num_topics,
    seed,
    iterations,
    alpha,
    eta,
    on_progress=None,  # callable(step:int, total:int, ll:float)
):
    set_global_seeds(seed)
    model = DMRModel(k=num_topics, seed=seed, alpha=alpha, eta=eta, tw=TermWeight.ONE)

    corpus = utils.Corpus()
    for idx, (grid, line) in enumerate(zip(gridL, lineL)):
        if not isinstance(line, str):
            raise TypeError(f"lineL[{idx}] must be str, got {type(line).__name__}")
        tokens = normalize_whitespace(line).split(" ")
        corpus.add_doc(tokens, a_data=grid)
    model.add_corpus(corpus, transform=transform_a_data_to_metadata)

    for i in range(0, iterations, 20):
        model.train(20)  # default workers; matches Colab
        if on_progress:
            try:
                on_progress(i + 20, iterations, float(model.ll_per_word))
            except Exception:
                pass

    topics = [model.get_topic_words(i, top_n=20) for i in range(model.k)]
    doc_dists = [doc.get_topic_dist() for doc in model.docs]
    dt_df = pd.DataFrame(
        doc_dists, columns=[f"T{i}" for i in range(num_topics)], index=range(len(doc_dists))
    )
    return model, topics, dt_df


# --- Streamlit UI ---
def show():
    # --- Part 1: Text similarity ---
    with st.expander("파트 1: 일반 텍스트 벡터화 및 유사도 분석", expanded=True):
        st.subheader("1. 분석할 문서 입력")
        default_texts = (
            "저는 사과 좋아요
"
            "저는 바나나 좋아요
"
            "저는 바나나 좋아요 저는 바나나 좋아요"
        )
        text_input = st.text_area(
            "분석할 문장들을 한 줄에 하나씩 입력하세요.",
            value=default_texts,
            height=100,
            key="sim_text_input",
        )
        texts = [line.strip() for line in text_input.split("
") if line.strip()]

        if texts:
            doc_labels = [f"문서{i+1}" for i in range(len(texts))]
            st.subheader("2. 단순 빈도(Count) 기반 분석")
            if st.button("단순 빈도 DTM 및 유사도 분석 실행", type="primary"):
                count_vec = CountVectorizer()
                X_count = count_vec.fit_transform(texts)
                counts = X_count.sum(axis=0).A1
                terms = count_vec.get_feature_names_out()
                df_tf = (
                    pd.DataFrame({"term": terms, "frequency": counts})
                    .sort_values(by=["frequency", "term"], ascending=[False, True])
                )
                sorted_terms = df_tf["term"].tolist()
                dtm_df = pd.DataFrame(
                    X_count.toarray(), columns=terms, index=doc_labels
                )[sorted_terms]
                st.session_state.count_dtm_df = dtm_df
                cos_sim = cosine_similarity(dtm_df)
                st.session_state.cos_sim_df = pd.DataFrame(
                    cos_sim, index=doc_labels, columns=doc_labels
                )
                st.session_state.tfidf_dtm_df = None
                st.session_state.tfidf_cos_sim_df = None

            if st.session_state.get("count_dtm_df") is not None:
                st.write("#### 문서-단어 행렬 (DTM)")
                st.dataframe(st.session_state.count_dtm_df, use_container_width=True)
                st.write("#### 각 문서의 특성 벡터")
                for i, row in st.session_state.count_dtm_df.iterrows():
                    st.text(f"{i}: {list(row)}")
                st.write("#### 문서 간 코사인 유사도")
                st.dataframe(st.session_state.cos_sim_df, use_container_width=True)

            st.subheader("3. 벡터 방식 비교: TF-IDF")
            if st.button("TF-IDF DTM 및 유사도 분석 실행"):
                tfidf_vec = TfidfVectorizer()
                X_tfidf = tfidf_vec.fit_transform(texts)
                scores = X_tfidf.sum(axis=0).A1
                terms = tfidf_vec.get_feature_names_out()
                df_tfidf = (
                    pd.DataFrame({"term": terms, "score_sum": scores})
                    .sort_values(by=["score_sum", "term"], ascending=[False, True])
                )
                sorted_terms = df_tfidf["term"].tolist()
                tfidf_dtm_df = pd.DataFrame(
                    X_tfidf.toarray(), columns=terms, index=doc_labels
                )[sorted_terms]
                st.session_state.tfidf_dtm_df = tfidf_dtm_df
                cos_sim_tfidf = cosine_similarity(tfidf_dtm_df)
                st.session_state.tfidf_cos_sim_df = pd.DataFrame(
                    cos_sim_tfidf, index=doc_labels, columns=doc_labels
                )

            if st.session_state.get("tfidf_dtm_df") is not None:
                st.write("#### 문서-단어 행렬 (DTM) - TF-IDF")
                st.dataframe(st.session_state.tfidf_dtm_df, use_container_width=True)
                st.write("#### 문서 간 코사인 유사도 (TF-IDF 기반)")
                st.dataframe(st.session_state.tfidf_cos_sim_df, use_container_width=True)

    st.divider()

    # --- Part 2: Topic features ---
    st.header("4. 개벽의 특성 벡터 추출")
    st.info(
        "1, 2단계에서 생성/전처리한 데이터 또는 유사한 구조의 파일을 업로드하여 텍스트의 특성을 추출합니다."
    )

    uploaded_file_tab4 = st.file_uploader(
        "특성 추출에 사용할 엑셀 파일을 업로드하세요 (doc_split_12gram, grid_1 열 필요).",
        type=["xlsx", "xls"],
        key="tab4_file_uploader",
    )

    if uploaded_file_tab4:
        df_for_feature = pd.read_excel(uploaded_file_tab4)

        with st.expander("4.1. '고빈도 단어'를 사용하는 경우", expanded=True):
            stopw_input = st.text_input(
                "제외할 단어(Stopwords)를 쉼표(,)로 구분하여 입력하세요.", value="문제,금일,관계"
            )
            rank_n_input = st.number_input(
                "추출할 고빈도 단어 개수를 입력하세요.",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
            )
            if st.button("고빈도 단어 특성 추출 실행"):
                stopw = [word.strip() for word in stopw_input.split(",")]
                with st.spinner("DTM 생성 및 고빈도 단어 추출 중..."):
                    st.session_state.highword_df = get_dtm_gb(
                        df_for_feature, "doc_split_12gram", stopw, rank_n_input
                    )
            if st.session_state.get("highword_df") is not None:
                st.success(f"상위 {rank_n_input}개 고빈도 단어 DTM이 생성되었습니다.")
                st.dataframe(st.session_state.highword_df, use_container_width=True)

        with st.expander("4.2. '토픽'을 사용하는 경우", expanded=True):
            st.info("Tomotopy DMR 모델을 사용하여 문서 집합의 주요 '주제(Topic)'를 찾아냅니다.")
            st.write("**모델 파라미터 설정**")
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                num_topics_input = st.number_input("토픽 개수 (k)", min_value=2, max_value=50, value=4)
                alpha_input = st.number_input(
                    "알파 (alpha)", min_value=0.01, max_value=1.0, value=0.05, step=0.01, format="%.2f"
                )
            with p_col2:
                iterations_input = st.number_input(
                    "반복 횟수 (iterations)", min_value=100, max_value=5000, value=1000, step=100
                )
                eta_input = st.number_input(
                    "에타 (eta)", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f"
                )
            with p_col3:
                seed_input = st.number_input("랜덤 시드 (seed)", value=103)

            if st.button("토픽 모델링 특성 추출 실행"):
                status_placeholder = st.empty()
                gridL = df_for_feature["grid_1"].to_list()
                lineL = df_for_feature["doc_split_12gram"].to_list()

                progress_bar = st.progress(0)
                log_area = st.empty()
                log_buf = []

                def _on_progress(step: int, total: int, ll: float) -> None:
                    log_buf.append(f"Iteration: {step}/{total}	Log-likelihood: {ll:.4f}")
                    log_area.code("
".join(log_buf))
                    progress_bar.progress(min(step / total, 1.0))

                try:
                    model, topics, dt_df = run_dmr_model(
                        gridL=gridL,
                        lineL=lineL,
                        num_topics=num_topics_input,
                        seed=seed_input,
                        iterations=iterations_input,
                        alpha=alpha_input,
                        eta=eta_input,
                        on_progress=_on_progress,
                    )
                    st.session_state.topic_df = dt_df
                    st.session_state.topics_words = topics
                    status_placeholder.success("토픽 모델링이 완료되었습니다!")
                except Exception as e:
                    st.exception(e)
                    return

            if st.session_state.get("topic_df") is not None:
                st.write("**추출된 토픽별 주요 단어:**")
                topic_word_df = pd.DataFrame(
                    [[word[0] for word in topic] for topic in st.session_state.topics_words],
                    index=[f"Topic {i}" for i in range(len(st.session_state.topics_words))],
                ).transpose()
                st.dataframe(topic_word_df, use_container_width=True)
                st.write("**문서별 토픽 분포 (특성 벡터):**")
                st.dataframe(st.session_state.topic_df, use_container_width=True)
# --- END BLOCK-FEATURE ---
