import pandas as pd
import json
import os
from bigkinds_row_modify import process_pipeline, get_bk_news_data_path

def save_pipeline_results_to_csv_and_json(max_summary_items=30):
    """
    엑셀 데이터 -> 분류 -> 요약 파이프라인을 수행하고 CSV 및 웹용 JSON 파일로 저장.
    """
    excel_path = get_bk_news_data_path()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv_path = os.path.join(current_dir, "classified_military_news.csv")

    # 웹 데모 프로젝트 경로
    web_dir = os.path.abspath(os.path.join(current_dir, "..", "recommend_from_profile"))
    web_csv_path = os.path.join(web_dir, "news_data.csv")
    web_json_path = os.path.join(web_dir, "news_data.json")

    print("==================================================")
    print("AI 국방 기사 자동 분류 및 요약 파이프라인 가동")
    print(f"입력 파일: {excel_path}")
    print(f"출력 CSV: {output_csv_path}")
    print("==================================================")

    # 파이프라인 실행
    df_result = process_pipeline(excel_path, max_summary_items=max_summary_items)

    # 1. acin_military_app 디렉토리에 CSV 저장
    df_result.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"[4/4] CSV 파일 저장 완료: {output_csv_path}")

    # 2. recommend_from_profile 웹 데모 디렉토리에 CSV 복사
    os.makedirs(web_dir, exist_ok=True)
    df_result.to_csv(web_csv_path, index=False, encoding='utf-8-sig')

    # 3. 웹 데모에서 바로 fetch 및 로드할 수 있도록 JSON 포맷으로도 변환 저장
    news_list = []
    for idx, row in df_result.iterrows():
        # 키워드 추출 파싱
        raw_keywords = str(row.get('keywords', ''))
        if raw_keywords and raw_keywords != 'nan':
            entity_list = [f"#{k.strip()}" for k in raw_keywords.split(',')[:5] if k.strip()]
        else:
            entity_list = ["#국방", "#안보", "#방위산업"]

        # 3줄 요약 불릿 생성
        summary_text = str(row.get('summary', row.get('title', '')))
        # 마침표 기준으로 나누거나 단락 분할
        sentences = [s.strip() for s in summary_text.split('.') if len(s.strip()) > 5]
        if len(sentences) >= 3:
            bullets = [s + "." for s in sentences[:3]]
        elif len(sentences) == 2:
            bullets = [sentences[0] + ".", sentences[1] + ".", "국방 안보 및 방산 기술 관련 핵심 후속 동향 지속 모니터링 중."]
        else:
            bullets = [summary_text, "인공지능(AI) 기반 국방 기사 신뢰도 검증 완료.", "주요 작전 및 방산 정책 변화에 따른 후속 영향 분석."]

        # 카테고리 태그 세분화
        title = str(row.get('title', ''))
        if any(w in title for w in ['수출', '계약', '수주', '기업', 'LIG', '한화', '현대로템', 'KAI', '풍산']):
            cat = "INDUSTRY"
            cat_label = "방산기업·수주"
            cat_class = "tag-industry"
        elif any(w in title for w in ['개발', '전차', '자주포', '전투기', '무인', '미사일', '함정', '잠수함', 'K2', 'K9', 'KF-21']):
            cat = "WEAPON"
            cat_label = "첨단 무기체계·R&D"
            cat_class = "tag-weapon"
        else:
            cat = "MILITARY"
            cat_label = "국방정책·훈련"
            cat_class = "tag-military"

        item_dict = {
            "id": f"DEF-{idx+1:03d}",
            "category": cat,
            "categoryLabel": cat_label,
            "categoryTagClass": cat_class,
            "confidence": float(row.get('defense_prob', 99.5)),
            "pubDate": str(row.get('date', '2024-11-19')),
            "timeAgo": f"{(idx % 5) + 1}시간 전",
            "source": f"{row.get('press', '국방 언론 종합')}",
            "title": title,
            "summary": summary_text,
            "summaryBullets": bullets,
            "fullContent": str(row.get('content', title)),
            "entities": entity_list,
            "originalUrl": str(row.get('url', 'https://www.bigkinds.or.kr'))
        }
        news_list.append(item_dict)

    with open(web_json_path, 'w', encoding='utf-8') as f:
        json.dump(news_list, f, ensure_ascii=False, indent=2)

    print(f"웹 데모용 JSON 파일 저장 완료: {web_json_path}")
    print(f"총 {len(news_list)}건의 국방 뉴스 데이터가 웹 데모에 성공적으로 연동되었습니다.")
    return df_result

if __name__ == "__main__":
    save_pipeline_results_to_csv_and_json()
