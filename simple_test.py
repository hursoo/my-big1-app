# simple_test.py

import tomotopy as tp
import numpy as np
import warnings

# RuntimeWarning 무시
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_test():
    """
    tomotopy와 numpy의 호환성 검증을 위해
    최소한의 LDA 모델 학습 및 Log-likelihood 계산을 실행합니다.
    """
    # 1. 실제 'gb_df['doc_split_12gram'].to_list()'와 유사한 구조의 샘플 데이터 생성
    print("샘플 코퍼스 데이터를 생성합니다...")
    corpus = [
        ['역사', '데이터', '분석', '토픽', '모델링'],
        ['파이썬', '코딩', '재미있어요', '스트림릿'],
        ['버전', '충돌', '해결', '어려워요'],
        ['나는', '사과', '바나나', '좋아해요'],
        ['오늘', '날씨', '정말', '좋아요']
    ]

    # 2. 빠른 테스트를 위한 파라미터 설정
    k_test = 5  # 단일 k 값으로 테스트
    seed = 100
    train_steps = 100 # 빠른 테스트를 위해 학습 횟수 감소

    try:
        # 3. LDA 모델 초기화 및 학습 (사용자 코드 기반)
        print(f"\nk={k_test}로 LDA 모델 초기화를 시작합니다...")
        mdl = tp.LDAModel(k=k_test, seed=seed)

        print("모델에 문서를 추가합니다...")
        for doc in corpus:
            mdl.add_doc(doc)

        print(f"{train_steps}회 학습을 진행합니다...")
        mdl.burn_in = 10
        mdl.train(train_steps, workers=1)
        print("모델 학습이 완료되었습니다.")

        # 4. Log-likelihood 계산 (오류 발생 지점 확인)
        print("Log-likelihood를 계산합니다...")
        log_likelihood = mdl.ll_per_word
        print("계산에 성공했습니다.")

        # 5. 최종 결과 출력
        print("\n--- ✨ 테스트 결과 ---")
        print(f"  - Tomotopy Version: {tp.__version__}")
        print(f"  - Numpy Version: {np.__version__}")
        print(f"  - Log-likelihood (k={k_test}): {log_likelihood:.4f}")
        print("\n✅ 성공: 현재 환경의 tomotopy와 numpy는 완벽하게 호환됩니다!")

    except Exception as e:
        print("\n--- ❌ 테스트 실패 ---")
        print(f"오류가 발생했습니다: {e}")
        print("\n패키지 설치에 여전히 문제가 있을 수 있습니다.")

# --- 테스트 실행 ---
if __name__ == "__main__":
    run_test()