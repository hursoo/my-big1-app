import streamlit as st
import pandas as pd
import ssl
from urllib.request import urlopen, Request
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
    # 일부 웹사이트의 접근 거부를 피하기 위한 헤더 정보
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i, url in enumerate(urls[:n]):
        try:
            # 헤더를 포함하여 요청(Request) 객체를 생성
            req = Request(url, headers=headers)
            webpage = urlopen(req, context=ctx)
            
            r_id = url[-16:]
            
            # 파이썬 기본 파서 'html.parser' 사용
            bsobj = BeautifulSoup(webpage.read(), 'html.parser')
            
            # id가 'cont_view'인 div를 찾음
            content_div = bsobj.find('div', {'id': 'cont_view'})
            
            # div를 실제로 찾았는지 확인 (가장 중요한 안전장치)
            if content_div:
                text_content = content_div.get_text('\n', strip=True)
                results.append([r_id, text_content])
            else:
                st.warning(f"'{url}'에서 콘텐츠 영역('cont_view')을 찾지 못했습니다. 이 URL은 건너뜁니다.")
                
        except Exception as e:
            st.warning(f"'{url}' 스크래핑 중 오류 발생: {e}")
            continue
        
        progress_bar.progress((i + 1) / n, text=f"스크래핑 진행 중... ({i+1}/{n})")
    
    progress_bar.empty()
    if not results:
        return pd.DataFrame() # 결과가 없으면 빈 데이터프레임 반환
    return pd.DataFrame(results, columns=['r_id', 'content'])

def show():
    st.header("자료 불러오기 및 스크래핑: gb_011_Scraping(big1).ipynb")
    st.info("[input] gb_data_2.1.xlsx, 근현대잡지자료_20250315172708.txt")
    st.info("[output] ron10_data.xlsx")

    col1, col2 = st.columns(2)
    ron_info_df = None
    gisa_info_df = None

    with col1:
        excel_file = st.file_uploader("gb_data_2.1.xlsx(`ron_info` 시트 포함)", type=['xlsx', 'xls'], key="ws_excel_uploader")
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
        gisa_file = st.file_uploader("기사 정보파일(근현대잡지자료_20250315172708.txt)", type="txt", key="ws_gisa_uploader")
        if gisa_file:
            try:
                gisa_info_df = pd.read_csv(gisa_file, sep='^', encoding='utf-8')
                st.dataframe(gisa_info_df.head(2), use_container_width=True)
            except Exception as e:
                st.error(f"텍스트 파일 처리 중 오류: {e}")

    if ron_info_df is not None:
        st.divider()
        st.subheader("스크래핑 실행")
        urls = ron_info_df['url'].tolist()
        num_to_scrape = st.slider("스크래핑할 논설 개수", 1, len(urls), min(10, len(urls)), key="ws_slider")

        if st.button("웹 스크래핑 및 데이터 결합 시작", type="primary", key="ws_start_button"):
            if gisa_info_df is None:
                st.error("기사 정보(txt) 파일도 함께 업로드해야 합니다.")
            else:
                with st.spinner("작업을 처리 중입니다..."):
                    contents_df = get_contents(urls, num_to_scrape)
                    
                    if not contents_df.empty and 'r_id_raw' in ron_info_df.columns:
                        r334_info1 = ron_info_df.drop('r_id', axis=1, errors='ignore')
                        combi_df = pd.merge(r334_info1, contents_df, left_on='r_id_raw', right_on='r_id', how='inner')
                        combi_df1 = combi_df[['r_id', 'r_id_raw', 'title', 'writer', 'gisa_class', 'date', 'url', 'year', 'content']]
                        st.session_state.scraped_df = combi_df1
                    elif contents_df.empty:
                        st.warning("스크래핑된 콘텐츠가 없습니다. URL을 확인하거나 다른 방법으로 진행해주세요.")
                    else:
                        st.error("'ron_info_df'에 'r_id_raw' 열이 없습니다. 엑셀 파일을 확인해주세요.")
    
    st.divider()

    if 'scraped_df' in st.session_state and st.session_state.scraped_df is not None:
        st.subheader("작업 결과")
        st.success("스크래핑 및 결합 작업이 완료되었습니다.")
        st.dataframe(st.session_state.scraped_df, use_container_width=True)
        
        excel_data = convert_df_to_excel(st.session_state.scraped_df)
        st.download_button(
            "결과 다운로드 (Excel)",
            excel_data,
            'ron10_data.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="ws_download_button"
        )
        
        if st.button("🔄 결과 지우고 새로 시작하기", key="ws_reset_button"):
            if 'scraped_df' in st.session_state:
                del st.session_state.scraped_df
            st.rerun()

