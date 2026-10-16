import pandas as pd
import os
from classification_title import run
from summary_text import newSum

# 0. Get Excel file path
def get_bk_news_data_path():
    """
    로컬 빅카인즈 뉴스 엑셀 파일 경로 반환.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "NewsResult_20241119-20241119.xlsx")
    return file_path

# 1. Extract required columns: 제목, 본문, URL, 언론사, 일자, 키워드 등
def extract_title_text_url(file_path):
    print(f"[1/4] 엑셀 파일 로딩 중: {file_path}")
    df = pd.read_excel(file_path)
    print(f"전체 기사 건수: {len(df)}건")

    # 필요한 컬럼 추출 및 정리
    # 빅카인즈 표준 컬럼: ['제목', '본문', 'URL', '언론사', '일자', '특성추출(가중치순 상위 50개)', '통합 분류1']
    rename_cols = {
        '제목': 'title',
        '본문': 'content',
        'URL': 'url',
        '언론사': 'press',
        '일자': 'date',
        '특성추출(가중치순 상위 50개)': 'keywords',
        '통합 분류1': 'category'
    }

    available_cols = [col for col in rename_cols.keys() if col in df.columns]
    df_extracted = df[available_cols].copy()
    df_extracted = df_extracted.rename(columns=rename_cols)

    # 결측치 처리
    df_extracted['title'] = df_extracted['title'].fillna('')
    df_extracted['content'] = df_extracted['content'].fillna('')
    df_extracted['url'] = df_extracted['url'].fillna('')
    if 'press' not in df_extracted.columns:
        df_extracted['press'] = '언론사 종합'
    if 'date' not in df_extracted.columns:
        df_extracted['date'] = '2024-11-19'
    if 'keywords' not in df_extracted.columns:
        df_extracted['keywords'] = ''

    return df_extracted

# 2. Classification titles using KLUE-BERT
def classify_news_titles(input_df):
    print(f"[2/4] KLUE-BERT 국방 기사 분류 시작 ({len(input_df)}건)...")
    # classification_title.run() 호출
    classified_df = run(input_df)
    print(f"국방 기사(Label=1) 분류 완료: {len(classified_df)}건 검출")
    return classified_df

# 3. Summarize news contents using KoBART
def summarize_news(df_defense, max_items=50):
    """
    분류된 국방 기사의 본문을 KoBART 모델로 요약.
    실행 시간 및 API 자원을 고려하여 지정된 개수(기본 상위 50개)를 정밀 요약.
    """
    target_df = df_defense.head(max_items).copy()
    print(f"[3/4] KoBART 국방 기사 본문 요약 시작 ({len(target_df)}건)...")

    summaries = []
    total = len(target_df)

    for idx, (i, row) in enumerate(target_df.iterrows()):
        content = str(row['content']).strip()
        title = str(row['title']).strip()

        if len(content) > 30:
            try:
                # 텍스트가 너무 길면 앞 1000자 기준 요약
                trunc_text = content[:1000]
                summary = newSum(trunc_text)
                # 요약이 너무 짧거나 비어있을 경우 보정
                if not summary or len(summary) < 15:
                    summary = content[:200] + "..."
            except Exception as e:
                print(f"[{idx+1}/{total}] 요약 중 오류 발생 (대체 요약 적용): {e}")
                summary = content[:200] + "..."
        else:
            summary = title

        summaries.append(summary)
        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            print(f"  요약 진행률: {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")

    target_df['summary'] = summaries
    return target_df

# 4. Full Pipeline Execution
def process_pipeline(file_path=None, max_summary_items=30):
    if file_path is None:
        file_path = get_bk_news_data_path()

    # 1. 엑셀 데이터 추출
    df_raw = extract_title_text_url(file_path)

    # 2. KLUE-BERT 분류
    df_defense = classify_news_titles(df_raw)

    # 3. 본문 요약 생성
    df_summarized = summarize_news(df_defense, max_items=max_summary_items)

    return df_summarized
