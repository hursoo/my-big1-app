import streamlit as st
import pandas as pd

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

def show(xls: pd.ExcelFile):
    st.header("🖇️ 시트 및 열 결합 (Merge)")

    if xls is None:
        st.warning("데이터를 결합하려면 먼저 사이드바에서 엑셀 파일을 로드해주세요.")
        return

    # --- 세션 상태 초기화 ---
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    if 'merge_history' not in st.session_state:
        st.session_state.merge_history = []
    
    # --- 리셋 버튼 ---
    if st.button("🔄 결합 작업 모두 리셋하기"):
        st.session_state.merged_df = None
        st.session_state.merge_history = []
        if 'final_df_for_analysis' in st.session_state:
            del st.session_state.final_df_for_analysis
        st.rerun()

    # --- 1. 데이터 결합 단계 ---
    st.subheader("1단계: 시트 선택 및 결합")
    
    # --- 베이스(Left) 데이터 설정 ---
    if st.session_state.merged_df is None:
        # ✨ 변경점 1: 기본(Left) 시트를 첫 번째 시트로 자동 선택하고 미리보기 제공
        st.write("결합의 기준이 될 **기본(Left) 시트**를 선택하세요. (첫 번째 시트가 기본으로 선택됩니다.)")
        
        # selectbox의 index를 0으로 설정하여 첫 번째 시트를 기본 선택
        first_sheet_name = xls.sheet_names[0]
        left_sheet_name = st.selectbox(
            "기본 시트", 
            options=xls.sheet_names, 
            index=0, 
            key="left_sheet", 
            label_visibility="collapsed"
        )
        
        base_df = pd.read_excel(xls, sheet_name=left_sheet_name)
        st.write(f"**'{left_sheet_name}' 시트 미리보기:**")
        st.dataframe(base_df.head(), use_container_width=True)

    else:
        st.write("현재까지 결합된 데이터입니다. 여기에 추가로 결합합니다.")
        base_df = st.session_state.merged_df
        st.info(f"결합된 데이터 구조: **{base_df.shape[0]}행, {base_df.shape[1]}열** | 내역: {' → '.join(st.session_state.merge_history)}")
        st.dataframe(base_df.head(3), use_container_width=True)

    # --- 이하 코드는 동일 ---
    st.write("추가로 결합할 **추가(Right) 시트**와 기준(Key) 열을 선택하세요.")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        right_sheet_name = st.selectbox("추가할 시트", options=xls.sheet_names, key="right_sheet", label_visibility="collapsed")
        right_df = pd.read_excel(xls, sheet_name=right_sheet_name)
    with col2:
        how_to_merge = st.selectbox("결합 방식", options=["inner", "left", "outer", "right"], help="`inner`: 양쪽에 모두 키가 있는 행만 유지, `left`: 왼쪽 기준", label_visibility="collapsed")

    key_col1, key_col2 = st.columns(2)
    with key_col1:
        left_key = st.selectbox("Left Key (현재 데이터의 기준 열)", options=base_df.columns.tolist(), key="left_key")
    with key_col2:
        right_key = st.selectbox("Right Key (추가할 시트의 기준 열)", options=right_df.columns.tolist(), key="right_key")

    if st.button("선택한 시트 결합하기", type="primary"):
        try:
            if st.session_state.merged_df is None:
                merged = pd.merge(base_df, right_df, left_on=left_key, right_on=right_key, how=how_to_merge)
                st.session_state.merge_history = [left_sheet_name, right_sheet_name]
            else:
                merged = pd.merge(st.session_state.merged_df, right_df, left_on=left_key, right_on=right_key, how=how_to_merge)
                st.session_state.merge_history.append(right_sheet_name)
            
            st.session_state.merged_df = merged
            st.rerun()
        except Exception as e:
            st.error(f"결합 중 오류: {e}")

    if st.session_state.merged_df is not None:
        st.divider()
        st.subheader("2단계: 결과 확인 및 추가 작업")

        with st.expander("결합 결과에서 열(Column) 제거하기 (선택 사항)"):
            cols_to_drop = st.multiselect("제거할 열을 선택하세요.", options=st.session_state.merged_df.columns.tolist())
            if st.button("선택한 열 제거"):
                st.session_state.merged_df = st.session_state.merged_df.drop(columns=cols_to_drop)
                st.rerun()

        st.write("현재 결과에 만족하시면 아래에서 최종 열을 선택하고 다운로드할 수 있습니다. 추가 결합을 원하시면 **위 1단계**를 계속 진행하세요.")
        
        st.subheader("3단계: 최종 데이터 확정 및 다운로드")
        final_cols = st.multiselect(
            "최종 데이터에 포함할 열을 모두 선택하세요.",
            options=st.session_state.merged_df.columns.tolist(),
            default=st.session_state.merged_df.columns.tolist()
        )
        
        final_df = st.session_state.merged_df[final_cols]
        st.dataframe(final_df, use_container_width=True)
        
        st.session_state.final_df_for_analysis = final_df
        
        st.download_button("결합된 데이터 다운로드 (CSV)", convert_df_to_csv(final_df), 'merged_data.csv', 'text/csv')