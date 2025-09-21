import streamlit as st
import pandas as pd
from collections import Counter

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def integrate_sentences(df):
    """5개 문장씩 묶어서 문서 단위로 통합하는 함수"""
    result = []
    doc_id_counter = 1

    # 기사 단위로 처리 (r_no, ho_no 기준 그룹)
    for (r_no, ho_no), group in df.groupby(['r_no', 'ho_no']):
        sent_raw_list = group['sent_raw'].tolist()
        sent_split_list = group['sent_split'].tolist()

        integrated_raw = []
        integrated_split = []
        temp_raw = []
        temp_split = []

        for raw, split in zip(sent_raw_list, sent_split_list):
            temp_raw.append(raw)
            temp_split.append(split)

            # 5개씩 묶기
            if len(temp_raw) == 5:
                integrated_raw.append(temp_raw)
                integrated_split.append(temp_split)
                temp_raw = []
                temp_split = []

        # 자투리 문장 처리
        if len(temp_raw) > 0:
            if len(temp_raw) <= 2 and len(integrated_raw) > 0:  # 자투리 2개 이하
                integrated_raw[-1].extend(temp_raw)
                integrated_split[-1].extend(temp_split)
            else:  # 자투리 3개 이상
                integrated_raw.append(temp_raw)
                integrated_split.append(temp_split)

        # 결과 저장
        for raw_group, split_group in zip(integrated_raw, integrated_split):
            result.append({
                'doc_id': doc_id_counter,
                'doc_raw': " ".join(raw_group),
                'doc_split': " ".join(split_group),
                'r_no': r_no,
                'ho_no': ho_no
            })
            doc_id_counter += 1

    return pd.DataFrame(result)

def generate_bigrams(text):
    """2-gram 생성 함수"""
    words = text.split()
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append("_".join([words[i], words[i + 1]]))
    return bigrams

def insert_bigrams(text, filtered_bigrams):
    """절삭된 2-gram을 텍스트에 삽입하는 함수"""
    words = text.split()
    new_words = []
    for i in range(len(words) - 1):
        bigram = "_".join([words[i], words[i + 1]])
        new_words.append(words[i])
        if bigram in filtered_bigrams:
            new_words.append(bigram)
    new_words.append(words[-1])  # 마지막 단어 추가
    return " ".join(new_words)


def count_underscored_bigrams(column):
    """언더바 포함 단어의 빈도 계산 함수"""
    words = []
    for row in column:
        words.extend([word for word in row.split() if "_" in word])
    return Counter(words)

def filter_selected_bigrams_with_all_unigrams(text, selected_bigrams):
    """2-gram 중에서 선별된 단어만 남기는 함수"""
    words = text.split()
    filtered_words = [
        word for word in words
        if "_" not in word or word in selected_bigrams
    ]
    return " ".join(filtered_words)


# 기존 함수들에 추가
def remove_adjacent_words_with_bigram(row):
    """2-gram 앞뒤 단어 제거 함수"""
    words = row.split()
    new_words = words[:]

    i = 0
    while i < len(new_words):
        if "_" in new_words[i]:
            if new_words[i].count("_") == 1:
                left_word, right_word = new_words[i].split("_")

                if i > 0 and new_words[i - 1] == left_word:
                    new_words.pop(i - 1)
                    i -= 1

                if i < len(new_words) - 1 and new_words[i + 1] == right_word:
                    new_words.pop(i + 1)

        i += 1

    return " ".join(new_words)

def combine_to_basic_df(df, r, ho, writer):
    """개벽 데이터 정보 결합 -> 분석의 기본 df 생성"""
    df_r = pd.merge(df, r, left_on='r_no', right_on='r_id', how='inner')
    df_rho = pd.merge(df_r, ho, left_on='ho_no', right_on='ho_id', how='inner')
    df_rho_writer = pd.merge(df_rho, writer, left_on='w_new', right_on='w_new_id', how='inner')
    return_df = df_rho_writer[['doc_id', 'doc_raw', 'doc_split_12gram', 'r_no', 'title', 'w_new', 'ho_no', 'grid_1', 'wn_cls']]
    return return_df



def show(xls: pd.ExcelFile):
    st.header("🖇️ 개벽 데이터 결합 및 문서 통합")

    if xls is None:
        st.warning("데이터를 결합하려면 먼저 엑셀 파일을 로드해주세요.")
        return

    # --- 세션 상태 초기화 ---
    if 'final_df_for_analysis' not in st.session_state:
        st.session_state.final_df_for_analysis = None
    if 'integrated_df' not in st.session_state:
        st.session_state.integrated_df = None
    
    # --- 1. 데이터 불러오기 및 미리보기 ---
    st.subheader("1단계: 시트별 데이터 확인")
    
    try:
        # 각 시트 불러오기
        sent = pd.read_excel(xls, sheet_name='sent')
        r = pd.read_excel(xls, sheet_name='ron').drop('ho_no', axis=1)
        ho = pd.read_excel(xls, sheet_name='ho').drop('grid', axis=1)
        writer = pd.read_excel(xls, sheet_name='writer_new')
        
        # 각 시트 정보 표시
        col1, col2 = st.columns(2)
        with col1:
            st.write("**sent 시트:**", f"{sent.shape[0]}행, {sent.shape[1]}열")
            st.dataframe(sent.head(3), use_container_width=True)
            
            st.write("**ron 시트 (ho_no 제거 후):**", f"{r.shape[0]}행, {r.shape[1]}열")
            st.dataframe(r.head(3), use_container_width=True)
        
        with col2:
            st.write("**ho 시트 (grid 제거 후):**", f"{ho.shape[0]}행, {ho.shape[1]}열")
            st.dataframe(ho.head(3), use_container_width=True)
            
            st.write("**writer_new 시트:**", f"{writer.shape[0]}행, {writer.shape[1]}열")
            st.dataframe(writer.head(3), use_container_width=True)
            
    except Exception as e:
        st.error(f"시트를 불러오는 중 오류가 발생했습니다: {e}")
        return

    # --- 2. 데이터 처리 실행 ---
    st.subheader("2단계: 문장 통합 및 데이터 결합")

    if st.button("데이터 처리 실행", type="primary"):
        try:
            with st.spinner("데이터를 처리하는 중..."):
                
                # 1단계: 5개 문장씩 묶어서 문서 단위로 통합
                st.write("🔄 문장 통합 중 (5개씩 묶기)...")
                result_df = integrate_sentences(sent)
                st.success(f"✅ 문장 통합 완료: {result_df.shape[0]}개 문서 생성")
                
                # 2단계: 논설 정보 결합 (r_no 기준)
                st.write("🔄 논설 정보 결합 중...")
                merged_with_r = pd.merge(result_df, r, left_on='r_no', right_on='r_id', how='left')
                st.success(f"✅ 논설 정보 결합 완료: {merged_with_r.shape[0]}행, {merged_with_r.shape[1]}열")
                
                # 3단계: 호별 정보 결합 (ho_no 기준)
                st.write("🔄 호별 정보 결합 중...")
                merged_with_ho = pd.merge(merged_with_r, ho, left_on='ho_no', right_on='ho_id', how='left')
                st.success(f"✅ 호별 정보 결합 완료: {merged_with_ho.shape[0]}행, {merged_with_ho.shape[1]}열")
                
                # 4단계: 필요한 열만 선택
                st.write("🔄 최종 데이터 정리 중...")
                final_columns = ['doc_id', 'doc_raw', 'doc_split', 'r_no', 'ho_no']
                integrated_df = merged_with_ho[final_columns].copy()
                
                # 세션 상태에 저장
                st.session_state.integrated_df = integrated_df
                
                st.success(f"🎉 데이터 결합 완료!")
                st.success(f"**결합 결과**: {integrated_df.shape[0]}행, {integrated_df.shape[1]}열")
                
        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {e}")
            st.exception(e)

    # --- 2-1. 중간 결과 확인 ---
    if st.session_state.integrated_df is not None:
        st.divider()
        st.subheader("2-1단계: 중간 결과 확인")
        
        integrated_df = st.session_state.integrated_df
        
        # 기본 정보
        st.info(f"**중간 데이터**: {integrated_df.shape[0]}행, {integrated_df.shape[1]}열")
        st.write(f"**열 구성**: {', '.join(integrated_df.columns.tolist())}")
        
        # 미리보기
        st.write("**데이터 미리보기:**")
        st.dataframe(integrated_df.head(10), use_container_width=True)
        
       
        # 중간 결과 다운로드
        st.download_button(
            "📥 중간 결과 다운로드 (CSV)",
            convert_df_to_csv(integrated_df),
            'gb_integrated_data.csv',
            'text/csv'
        )

    ################

    # --- 3. 2-gram 생성 및 처리 ---
    if st.session_state.integrated_df is not None:
        st.divider()
        st.subheader("3단계: 2-gram 생성 및 처리")

        # --- 3-1. 기본적인 2-gram 생성 및 삽입 ---
        st.write("**3-1단계: 기본적인 2-gram 생성 및 삽입 (빈도 5 이상)**")
        
        threshold = st.number_input("2-gram 최소 빈도 (이하는 제거)", min_value=1, max_value=20, value=5)
        
        if st.button("3-1. 2-gram 기본 처리 실행", type="secondary"):
            try:
                with st.spinner("2-gram을 처리하는 중..."):
                    df = st.session_state.integrated_df.copy()
                    
                    st.write("🔄 2-gram 생성 및 빈도 계산 중...")
                    all_bigrams = []
                    for text in df["doc_split"]:
                        all_bigrams.extend(generate_bigrams(text))
                    
                    bigram_counts = Counter(all_bigrams)
                    st.write(f"전체 2-gram 종류: {len(bigram_counts)}개")
                    
                    filtered_bigrams = {k: v for k, v in bigram_counts.items() if v > threshold}
                    st.write(f"빈도 {threshold} 초과 2-gram: {len(filtered_bigrams)}개")
                    
                    st.write("🔄 2-gram 삽입 중...")
                    df["doc_split_updated"] = df["doc_split"].apply(
                        lambda x: insert_bigrams(x, filtered_bigrams)
                    )
                    
                    st.session_state.bigram_inserted_df = df
                    st.success(f"✅ 3-1단계 완료: {df.shape[0]}행, {df.shape[1]}열")
                    
            except Exception as e:
                st.error(f"3-1단계 처리 중 오류: {e}")

        # 3-1단계 결과 항상 표시
        if 'bigram_inserted_df' in st.session_state:
            st.write("**3-1단계 결과: doc_split_updated**")
            df = st.session_state.bigram_inserted_df
            st.write("**처음 5행:**")
            st.dataframe(df.head(5), use_container_width=True)
            st.write("**마지막 5행:**")
            st.dataframe(df.tail(5), use_container_width=True)

        # --- 3-2. 2-gram 선별 처리 ---
        if 'bigram_inserted_df' in st.session_state:
            st.write("**3-2단계: 2-gram 선별 처리 (빈도 30 이상, 어색한 조합 제외)**")
            
            if st.button("3-2. 2-gram 선별 처리 실행", type="secondary"):
                try:
                    with st.spinner("2-gram 선별 처리 중..."):
                        df1 = st.session_state.bigram_inserted_df[['doc_id', 'doc_raw', 'doc_split_updated', 'r_no', 'ho_no']].copy()
                        
                        st.write("🔄 2-gram 빈도 분석 중...")
                        bigram_frequencies = count_underscored_bigrams(df1["doc_split_updated"])
                        bigram_freq_df = pd.DataFrame(bigram_frequencies.items(), columns=["bigram", "frequency"])
                        bigram_freq_df = bigram_freq_df.sort_values(by="frequency", ascending=False).reset_index(drop=True)
                        
                        st.write("🔄 선별된 2-gram 리스트 구성 중...")
                        
                        top_2gram = ['朝鮮_人', '노동_者', '사회_主義', '자본_主義', '소작_人', '朝鮮_민족', '朝鮮_사람', 
                                '主義_者', '자본_家', '무산_계급', '사람_性', '사회_운동', '기독_敎', '朝鮮_사회', 
                                '帝國_主義', '中産_계급', '천도_敎', '無_정부', '농업_노동', '정부_主義', '사회_생활', 
                                '단체_생활', '일본_人', '계급_意識', '식민_地', '지식_계급', '노동_계급', '사회_문제', 
                                '민족_主義', '노동_문제', '인류_사회', '민족_개조', '사회_제도', '공산_主義', '문화_운동', 
                                '무산_者', '指導_者', '사람_自己', '일반_민중', '사회_조직', '인생_觀', '공산_黨', 
                                '민족_생활', '문제_해결', '민주_主義', '외국_人', '생산_力', '민족_감정', '일반_사회']
                        
                        selected_bigrams = [
                            "자유_평등", "어린_이", "마르크스_主義", "민족_性", "新_문화", "지주_소작", "자본_계급",
                            "계급_투쟁", "사회_현상", "정치_경제", "현대_문명", "생활_難", "중심_인물", "지배_계급",
                            "현대_사회", "생활_조건", "경제_정책", "인류_생활", "사상_家", "人道_正義", "朝鮮_민중",
                            "노동_운동", "농촌_문제", "자연_主義", "지배_者", "도착_點", "현대_人", "약소_민족",
                            "물산_장려", "프랑스_혁명", "專門_家", "생활_표준", "개인_主義", "계급_운동", "유산_者",
                            "자유_主義", "사상_문화", "당국_者", "정치_운동", "문화_생활", "중심_세력", "정치_家"
                        ]
                        
                        a = '종교_家 태평_洋 新_사회 不_완전 唯物_論 출발_點 하나_님 正_반대 不_합리 유럽_대전 민족_체면 사회_진화 사회_계급 생활_費 서양_人 인간_생활'
                        b = '소작_제도 사상_혁명 민족_운동 朝鮮_운동 중류_계급 생활_양식 경제_문제 생산_者 생산_관계 인류_구제 압박_민족 계급_지배 자연_과학 唯物_史觀 자본_제도 토지_소유'
                        c = '중추_계급 선교_師 婦人_문제 理想_主義 사회_개조 생활_향상 경제_생활 경제_조직 봉건_제도 볼세비키_主義 사회_경제 인간_사회 식민_정책 국가_민족'
                        d = '소작_料 사회_봉공 야만_人 人道_主義 무산_청년 예술_家 문화_건설 계급_대립 大_지주 사회_學 생활_문화 인류_主義 민중_운동 민중_운동 사회_생산 청년_운동'
                        e = '세계_개조 사회_黨 被_압박 부르주아_문화 문예_부흥 사회_性 부르주아_사회 군국_主義 소작_운동 유산_계급 소작_農 自作_農 가족_제도 민족_사회 朝鮮_농촌'
                        f = '경제_운동 공업_노동 實業_家 생활_문제 특권_계급 생활_維持 민족_중심 체면_維持 혁명_운동 동양_척식_회사 원동_力 정신_물질 보통_학교 러시아_혁명'
                        g = '총독_府 교육_문제 私有_재산 국제_연맹 被_정복 소비_者 청년_會 사회_혁명 종교_신앙 사회_건설 사회_사상 新_사상'
                        
                        combined_text = f"{a} {b} {c} {d} {e} {f} {g}"
                        word_list = combined_text.split()
                        ttl_bigram = top_2gram + selected_bigrams + word_list
                        
                        st.write("🔄 선별된 2-gram만 필터링 중...")
                        df1["doc_split_filtered"] = df1["doc_split_updated"].apply(
                            lambda x: filter_selected_bigrams_with_all_unigrams(x, ttl_bigram)
                        )
                        
                        st.session_state.filtered_df = df1
                        st.success(f"✅ 3-2단계 완료: {len(ttl_bigram)}개 2-gram 선별")
                        
                except Exception as e:
                    st.error(f"3-2단계 처리 중 오류: {e}")

            # 3-2단계 결과 항상 표시
            if 'filtered_df' in st.session_state:
                st.write("**3-2단계 결과: doc_split_filtered**")
                df1 = st.session_state.filtered_df
                st.write("**처음 5행:**")
                st.dataframe(df1.head(5), use_container_width=True)
                st.write("**마지막 5행:**")
                st.dataframe(df1.tail(5), use_container_width=True)

        # --- 3-3. 정리(앞뒤 단어 제거) ---
        if 'filtered_df' in st.session_state:
            st.write("**3-3단계: 정리 (앞뒤 단어 제거)**")
            
            if st.button("3-3. 앞뒤 단어 제거 실행", type="secondary"):
                try:
                    with st.spinner("앞뒤 단어 제거 중..."):
                        df1 = st.session_state.filtered_df.copy()
                        
                        st.write("🔄 2-gram 앞뒤 단어 삭제 중...")
                        df1["doc_split_filtered_1"] = df1["doc_split_filtered"].apply(remove_adjacent_words_with_bigram)
                        
                        df2 = df1[['doc_id', 'doc_raw', 'doc_split_filtered_1', 'r_no', 'ho_no']]
                        df3 = df2.rename(columns={'doc_split_filtered_1': 'doc_split_12gram'})
                        
                        st.session_state.final_processed_df = df3
                        st.success(f"✅ 3-3단계 완료: {df3.shape[0]}행, {df3.shape[1]}열")
                        
                except Exception as e:
                    st.error(f"3-3단계 처리 중 오류: {e}")

            # 3-3단계 결과 항상 표시
            if 'final_processed_df' in st.session_state:
                st.write("**3-3단계 결과:doc_split_12gram**")
                df3 = st.session_state.final_processed_df
                st.write("**처음 5행:**")
                st.dataframe(df3.head(5), use_container_width=True)
                st.write("**마지막 5행:**")
                st.dataframe(df3.tail(5), use_container_width=True)


    
    # --- 4. 구간 및 필자 정보 결합 ---
    if 'final_processed_df' in st.session_state:
        st.divider()
        st.subheader("4단계: 구간 및 필자 정보 결합")
        
        if st.button("4. 최종 결합 실행", type="primary"):
            try:
                with st.spinner("최종 결합 처리 중..."):
                    df3 = st.session_state.final_processed_df
                    
                    st.write("🔄 구간 및 필자 정보 결합 중...")
                    gb_df = combine_to_basic_df(df3, r, ho, writer)
                    
                    st.session_state.final_df_for_analysis = gb_df
                    st.success(f"🎉 모든 처리 완료!")
                    st.success(f"**최종 결과**: {gb_df.shape[0]}행, {gb_df.shape[1]}열")
                    
                    st.write("**최종 열 구성:**")
                    st.write(', '.join(gb_df.columns.tolist()))
                    
            except Exception as e:
                st.error(f"4단계 처리 중 오류: {e}")
                st.exception(e)

    # --- 5. 최종 결과 확인 및 다운로드 ---
    if st.session_state.final_df_for_analysis is not None:
        st.divider()
        st.subheader("5단계: 최종 결과 확인 및 다운로드")
        
        final_df = st.session_state.final_df_for_analysis
        
        st.info(f"**최종 데이터**: {final_df.shape[0]}행, {final_df.shape[1]}열")
        st.write(f"**열 구성**: {', '.join(final_df.columns.tolist())}")
        
        st.write("**최종 데이터 미리보기:**")
        st.dataframe(final_df.head(10), use_container_width=True)
        
        with st.expander("📊 최종 통계"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**필자별 문서 수:**")
                st.write(final_df['w_new'].value_counts().head(10))
            with col2:
                st.write("**필자 분류별 문서 수:**")
                st.write(final_df['wn_cls'].value_counts())
        
        st.download_button(
            "📥 최종 데이터 다운로드 (CSV)",
            convert_df_to_csv(final_df),
            'gb_data_2(doc,1g2g,wn_cls).csv',
            'text/csv'
        )