import streamlit as st
import pandas as pd

def show():
    """
    사이드바에 파일 업로드 및 URL 입력 UI를 표시하고,
    로드된 pandas ExcelFile 객체를 반환하는 함수.
    """
    st.sidebar.subheader("📂 파일 선택")
    input_method = st.sidebar.radio("입력 방식을 선택하세요:", ("URL", "파일 업로드"))

    file = None
    if input_method == "URL":
        url = st.sidebar.text_input("엑셀 파일의 URL을 입력하세요:", placeholder="URL을 여기에 붙여넣으세요.")
        if url: file = url
    else:
        uploaded_file = st.sidebar.file_uploader("엑셀 파일을 업로드하세요.", type=['xlsx', 'xls'])
        if uploaded_file: file = uploaded_file

    if file:
        try:
            return pd.ExcelFile(file)
        except Exception as e:
            st.sidebar.error(f"파일 로드 오류: {e}")
            return None
    return None