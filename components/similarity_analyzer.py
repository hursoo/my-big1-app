import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def show():
    # --- 1. 작업 개요 설명 ---
    st.header("텍스트 벡터화 및 유사도 분석")
    with st.expander("ℹ️ 이 작업은 무엇인가요?", expanded=True):
        st.write("""
        이 도구는 여러 텍스트(문서)를 수치화된 벡터로 변환하고, 이를 바탕으로 텍스트 간의 유사도를 측정합니다.
        먼저 단어의 단순 빈도를 기반으로 분석을 수행한 뒤, 추가로 TF-IDF 가중치 방식을 비교해볼 수 있습니다.
        """)

    # --- 세션 상태 초기화 ---
    if 'count_dtm_df' not in st.session_state: st.session_state.count_dtm_df = None
    if 'cos_sim_df' not in st.session_state: st.session_state.cos_sim_df = None
    if 'tfidf_dtm_df' not in st.session_state: st.session_state.tfidf_dtm_df = None
    if 'tfidf_cos_sim_df' not in st.session_state: st.session_state.tfidf_cos_sim_df = None

    # --- 2. 분석할 문서 입력 ---
    st.subheader("1. 분석할 문서 입력")
    default_texts = "저는 사과 좋아요\n저는 바나나 좋아요\n저는 바나나 좋아요 저는 바나나 좋아요"
    text_input = st.text_area(
        "분석할 문장들을 한 줄에 하나씩 입력하세요.", value=default_texts, height=100
    )
    texts = [line.strip() for line in text_input.split('\n') if line.strip()]

    if not texts:
        st.warning("분석할 문장을 한 줄 이상 입력해주세요.")
        return

    doc_labels = [f"문서{i+1}" for i in range(len(texts))]

    # --- 3. 단순 빈도(Count) 기반 분석 ---
    st.divider()
    st.subheader("2. 단순 빈도(Count) 기반 분석")
    if st.button("단순 빈도 DTM 및 유사도 분석 실행", type="primary"):
        try:
            count_vec = CountVectorizer()
            X_count = count_vec.fit_transform(texts)
            counts = X_count.sum(axis=0).A1
            terms = count_vec.get_feature_names_out()
            df_tf = pd.DataFrame({'term': terms, 'frequency': counts}).sort_values(by=['frequency', 'term'], ascending=[False, True])
            sorted_terms = df_tf['term'].tolist()
            
            # ✨ 변경점 1: Count DTM에 문서 인덱스 설정
            dtm_df = pd.DataFrame(X_count.toarray(), columns=terms, index=doc_labels)[sorted_terms]
            st.session_state.count_dtm_df = dtm_df
            
            cos_sim = cosine_similarity(dtm_df)
            st.session_state.cos_sim_df = pd.DataFrame(cos_sim, index=doc_labels, columns=doc_labels)
            
            st.session_state.tfidf_dtm_df = None
            st.session_state.tfidf_cos_sim_df = None
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

    if st.session_state.count_dtm_df is not None:
        st.write("#### 문서-단어 행렬 (DTM)")
        st.dataframe(st.session_state.count_dtm_df, use_container_width=True)
        st.write("#### 각 문서의 특성 벡터")
        for i, row in st.session_state.count_dtm_df.iterrows():
            st.text(f"{i}: {list(row)}")
        st.write("#### 문서 간 코사인 유사도")
        st.dataframe(st.session_state.cos_sim_df, use_container_width=True)

    # --- 4. 벡터 방식 비교: TF-IDF ---
    if st.session_state.count_dtm_df is not None:
        st.divider()
        st.subheader("3. 벡터 방식 비교: TF-IDF")
        
        if st.button("TF-IDF DTM 및 유사도 분석 실행"):
            try:
                tfidf_vec = TfidfVectorizer()
                X_tfidf = tfidf_vec.fit_transform(texts)
                scores = X_tfidf.sum(axis=0).A1
                terms = tfidf_vec.get_feature_names_out()
                df_tfidf = pd.DataFrame({'term': terms, 'score_sum': scores}).sort_values(by=['score_sum', 'term'], ascending=[False, True])
                sorted_terms = df_tfidf['term'].tolist()
                
                # ✨ 변경점 2: TF-IDF DTM에 문서 인덱스 설정
                tfidf_dtm_df = pd.DataFrame(X_tfidf.toarray(), columns=terms, index=doc_labels)[sorted_terms]
                st.session_state.tfidf_dtm_df = tfidf_dtm_df
                
                cos_sim_tfidf = cosine_similarity(tfidf_dtm_df)
                st.session_state.tfidf_cos_sim_df = pd.DataFrame(cos_sim_tfidf, index=doc_labels, columns=doc_labels)
                
            except Exception as e:
                st.error(f"TF-IDF 분석 중 오류: {e}")
        
        if st.session_state.tfidf_dtm_df is not None:
            st.write("#### 문서-단어 행렬 (DTM) - TF-IDF 가중치")
            st.dataframe(st.session_state.tfidf_dtm_df, use_container_width=True)
            
            if st.session_state.tfidf_cos_sim_df is not None:
                st.write("#### 문서 간 코사인 유사도 (TF-IDF 기반)")
                st.dataframe(st.session_state.tfidf_cos_sim_df, use_container_width=True)

    # --- 리셋 버튼 ---
    if st.button("🔄 이 작업 리셋하기"):
        st.session_state.count_dtm_df = None
        st.session_state.cos_sim_df = None
        st.session_state.tfidf_dtm_df = None
        st.session_state.tfidf_cos_sim_df = None
        st.rerun()