import time
import os
import sys
from save_data_csv import save_pipeline_results_to_csv_and_json
from bigkinds_row_modify import get_bk_news_data_path

def execute_pipeline_job():
    """
    빅카인즈 엑셀 파일(NewsResult_20241119-20241119.xlsx) 로드
    -> 제목 기반 국방 기사 분류(KLUE-BERT)
    -> 본문 텍스트 요약(KoBART)
    -> CSV 및 웹 데모용 JSON 파일 저장
    """
    try:
        excel_file = get_bk_news_data_path()
        if not os.path.exists(excel_file):
            print(f"[오류] 빅카인즈 엑셀 파일이 존재하지 않습니다: {excel_file}")
            return False

        print("\n" + "="*60)
        print(" [AI 국방 뉴스 파이프라인] 빅카인즈 데이터 처리 및 CSV 추출 시작")
        print("="*60)

        # 30개 국방 기사 요약 및 전체 국방 기사 CSV 저장
        save_pipeline_results_to_csv_and_json(max_summary_items=30)

        print("="*60)
        print(" [AI 국방 뉴스 파이프라인] 처리 및 CSV/JSON 저장 완료!")
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"[오류 발생] 파이프라인 실행 중 예외 발생: {e}")
        return False

def execute_periodically(interval_hours=6):
    """
    지정된 주기(기본 6시간)마다 파이프라인 자동 실행.
    """
    print(f"AI 국방 뉴스 자동 분석 데몬 시작 (업데이트 주기: {interval_hours}시간)")

    # 최초 1회 즉시 실행
    execute_pipeline_job()

    while True:
        wait_seconds = interval_hours * 60 * 60
        print(f"\n다음 업데이트까지 {interval_hours}시간 ({wait_seconds}초) 대기 중...")
        time.sleep(wait_seconds)
        execute_pipeline_job()

if __name__ == "__main__":
    # 단일 실행 모드
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        execute_periodically(6)
    else:
        execute_pipeline_job()