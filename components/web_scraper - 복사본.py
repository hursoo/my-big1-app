import streamlit as st
import pandas as pd
import ssl
from urllib.request import urlopen
from bs4 import BeautifulSoup
import io

@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def get_contents(urls, n):
    results = []
    progress_bar = st.progress(0, text="스크래핑 진행 중...")
    ctx = ssl._create_unverified_context()

    for i, url in enumerate(urls[:n]):
        try:
            webpage = urlopen(url, context=ctx)
            r_id = url[-16:]
            bsobj = BeautifulSoup(webpage.read(), 'lxml')
            List1 = bsobj.find_all('div', {'id': 'cont_view'})
            for z in List1:
                z1 = z.get_text('\n', strip=True)
                results.append([r_id, z1])
        except Exception as e:
            st.warning(f"{url} 스크래핑 중 오류 발생: {e}")
            continue
        
        progress_bar.progress((i + 1) / n, text=f"스크래핑 진행 중... ({i+1}/{n})")
    
    progress_bar.empty()
    return pd.DataFrame(results, columns=['r_id', 'content'])

def show():
    st.header("자료 불러오기 및 스크래핑")
    st.info("논설 정보 엑셀 파일과 기사 정보 텍스트 파일을 업로드하여 스크래핑을 시작하세요.")

    # --- 1. 파일 업로드 UI (항상 표시) ---
    col1, col2 = st.columns(2)
    ron_info_df = None
    with col1:
        excel_file = st.file_uploader("논설 정보(`ron_info` 시트 포함) 엑셀 파일", type=['xlsx', 'xls'])
        if excel_file:
            try:
                xls = pd.ExcelFile(excel_file)
                if 'ron_info' in xls.sheet_names:
                    ron_info_df = pd.read_excel(xls, sheet_name='ron_info')
                    st.dataframe(ron_info_df.head(2), use_container_width=True)
                else: 
                    st.error("'ron_info' 시트를 찾을 수 없습니다.")
            except Exception as e:
                st.error(f"엑셀 파일 처리 중 오류: {e}")

    with col2:
        gisa_file = st.file_uploader("기사 정보(`...txt`) 파일", type="txt")
        if gisa_file:
            try:
                gisa_info_df = pd.read_csv(gisa_file, sep='^', encoding='utf-8')
                st.dataframe(gisa_info_df.head(2), use_container_width=True)
            except Exception as e:
                st.error(f"텍스트 파일 처리 중 오류: {e}")

    # --- 2. 스크래핑 실행 UI (엑셀 로드 시 항상 표시) ---
    if ron_info_df is not None:
        st.divider()
        st.subheader("스크래핑 실행")
        urls = ron_info_df['url'].tolist()
        num_to_scrape = st.slider("스크래핑할 논설 개수", 1, len(urls), min(10, len(urls)))

        if st.button("웹 스크래핑 및 데이터 결합 시작", type="primary"):
            if gisa_file is None:
                st.error("기사 정보(txt) 파일도 함께 업로드해야 합니다.")
            else:
                with st.spinner("작업을 처리 중입니다..."):
                    # 스크래핑 및 결합 로직
                    contents_df = get_contents(urls, num_to_scrape)
                    r334_info1 = ron_info_df.drop('r_id', axis=1, errors='ignore')
                    combi_df = pd.merge(r334_info1, contents_df, left_on='r_id_raw', right_on='r_id', how='inner')
                    combi_df1 = combi_df[['r_id', 'r_id_raw', 'title', 'writer', 'gisa_class', 'date', 'url', 'year', 'content']]
                    # 결과를 세션 상태에 저장
                    st.session_state.scraped_df = combi_df1
    
    st.divider()

    # --- 3. 결과 표시 UI (세션에 결과가 있을 때 항상 표시) ---
    if 'scraped_df' in st.session_state and st.session_state.scraped_df is not None:
        st.subheader("작업 결과")
        st.success("스크래핑 및 결합 작업이 완료되었습니다.")
        st.dataframe(st.session_state.scraped_df, use_container_width=True)
        
        excel_data = convert_df_to_excel(st.session_state.scraped_df)
        st.download_button(
            "결과 다운로드 (Excel)",
            excel_data,
            'scraped_data.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        # 결과물을 지우고 다시 시작할 수 있는 리셋 버튼
        if st.button("🔄 결과 지우고 새로 시작하기"):
            del st.session_state.scraped_df
            st.rerun()