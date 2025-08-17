import streamlit as st
import pandas as pd
import re
from kiwipiepy import Kiwi

@st.cache_resource
def get_kiwi():
    return Kiwi()

def show(df_input):
    st.header("텍스트 전처리 파이프라인")

    # 세션 상태 초기화
    if 'sents_df' not in st.session_state: st.session_state.sents_df = None
    if 'tokenized_df' not in st.session_state: st.session_state.tokenized_df = None
    if 'final_df' not in st.session_state: st.session_state.final_df = None
    if 'target_tags' not in st.session_state: st.session_state.target_tags = ['SH', 'NP', 'NNP']

    # --- 1. 초기 데이터 및 열 선택 ---
    with st.expander("1. 전처리 대상 데이터 확인 및 열 선택", expanded=True):
        st.dataframe(df_input.head(3), use_container_width=True)
        default_cols = ['r_id', 'title', 'content']
        available_cols = [col for col in default_cols if col in df_input.columns]
        selected_cols = st.multiselect("분석에 사용할 열을 선택하세요.", options=df_input.columns, default=available_cols)
        df1 = df_input[selected_cols]
        st.write("선택된 데이터:")
        st.dataframe(df1.head(3), use_container_width=True)

    # --- 2. 문장 분리 ---
    if st.button("2. 문장 분리 실행"):
        with st.spinner("본문을 문장 단위로 분리 중..."):
            separators = [r'\n', r'\.', r'!', r'\?']
            pattern = '|'.join(separators)
            df1_copy = df1.copy()
            df1_copy['splited_sent'] = df1_copy['content'].apply(lambda x: re.split(pattern, str(x)))
            sents_df = df1_copy.explode('splited_sent').reset_index(drop=True)
            sents_df = sents_df[sents_df['splited_sent'].str.strip() != '']
            st.session_state.sents_df = sents_df
        st.rerun()

    if st.session_state.sents_df is not None:
        with st.expander("2. 문장 분리 결과 확인", expanded=True):
            st.dataframe(st.session_state.sents_df.head(), use_container_width=True)
            st.info(f"총 {len(st.session_state.sents_df)}개의 문장으로 분리되었습니다.")

    # --- 3. 형태소 분석 ---
    if st.session_state.sents_df is not None and st.button("3. 형태소 분석 실행 (Kiwi)"):
        with st.spinner("형태소 분석 중... (시간이 다소 소요될 수 있습니다)"):
            kiwi = get_kiwi()
            df = st.session_state.sents_df.copy()
            def simplify_kiwi(sent):
                analysis_result = kiwi.analyze(sent)
                if not analysis_result: return []
                return [(token.form + "/" + token.tag) for token in analysis_result[0][0]]
            df['tokens'] = df['splited_sent'].apply(simplify_kiwi)
            df = df[df['tokens'].apply(len) > 0]
            st.session_state.tokenized_df = df
        st.rerun()

    if st.session_state.tokenized_df is not None:
        with st.expander("3. 형태소 분석 결과 확인", expanded=True):
            st.dataframe(st.session_state.tokenized_df[['splited_sent', 'tokens']].head(), use_container_width=True)

    # --- 4. 특정 품사 추출 및 최종 정제 ---
    if st.session_state.tokenized_df is not None:
        st.subheader("4. 특정 품사 추출 및 최종 정제")
        st.session_state.target_tags = st.multiselect(
            "추출할 품사 태그를 선택하세요.",
            options=['NNG', 'NNP', 'NP', 'NR', 'SL', 'SH', 'SN', 'VV', 'VA', 'MAG'],
            default=st.session_state.target_tags
        )
        if st.button("4. 최종 정제 실행하기"):
            with st.spinner("최종 정제 중..."):
                df = st.session_state.tokenized_df.copy()
                df['morph'] = df['tokens'].apply(
                    lambda tokens: [token.split('/')[0] for token in tokens if token.split('/')[-1] in st.session_state.target_tags]
                )
                df = df[df['morph'].str.len() > 0]
                df['morph'] = df['morph'].apply(lambda x: ' '.join(x))
                df['tokens'] = df['tokens'].apply(lambda x: ' '.join(x))
                df = df.reset_index(drop=True)
                df['sent_id'] = df.index + 1
                
                # ✨ 변경점: 최종 결과물에 'tokens' 열을 다시 포함시킵니다.
                final_cols = ['sent_id', 'morph', 'splited_sent', 'tokens', 'r_id', 'title']
                final_df = df[[col for col in final_cols if col in df.columns]]
                st.session_state.final_df = final_df
            st.rerun()

    if st.session_state.final_df is not None:
        with st.expander("4. 최종 결과 확인 및 다운로드", expanded=True):
            st.dataframe(st.session_state.final_df.head(), use_container_width=True)
            st.info(f"최종 {len(st.session_state.final_df)}개의 전처리된 문장이 생성되었습니다.")
            from components.data_merger import convert_df_to_csv
            csv_data = convert_df_to_csv(st.session_state.final_df)
            st.download_button("전처리 결과 다운로드 (CSV)", csv_data, 'preprocessed_data.csv', 'text/csv')

    if st.button("🔄 전처리 작업 리셋하기"):
        st.session_state.sents_df = None
        st.session_state.tokenized_df = None
        st.session_state.final_df = None
        if 'target_tags' in st.session_state: del st.session_state.target_tags
        st.rerun()