from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from logger import logger
from middleware import trace_id_middleware
from security import verify_api_key

app = FastAPI(dependencies=[Depends(verify_api_key)])
#---------------------------------------------------
#로컬테스트용이라 테스트후 위에 것 살리기
#app = FastAPI()

app.middleware("http")(trace_id_middleware)

logger.info("FastAPI LangGraph 서버가 성공적으로 시작되었습니다!")


# ============================================================
# 요청 모델 (Spring → FastAPI)
# ============================================================

class PlaceCandidate(BaseModel):
    """Spring이 보내는 후보 장소"""
    id: str = Field(description="후보 ID. 'store-XXX' 또는 'culture-XXX' 형식")
    name: str
    address: str
    category: str = Field(default="", description="한식, 미술관/갤러리, 공연장 등")
    introduction: str | None = Field(default=None, description="짧은 소개")


class SpotCongestion(BaseModel):
    areaNm: str | None = None
    congestionLevel: str | None = None
    congestionLabel: str | None = None
    ppltnMin: int | None = None
    ppltnMax: int | None = None
    observedAt: str | None = None


class CourseRequest(BaseModel):
    date: str
    categories: list[str]
    members: str
    budget: str
    prompt: str
    # ★ 신규 필드: 후보 장소 리스트 (RAG 핵심)
    candidates: list[PlaceCandidate] = []
    congestionData: list[SpotCongestion] = []


# ============================================================
# 응답 모델 (FastAPI → Spring)
# ============================================================

class CourseResponse(BaseModel):
    """AI는 ID 리스트만 반환. 실제 장소 정보는 Spring에서 채움."""
    description: str = Field(
        description="코스 매력적 소개글 1~2줄. 사용자 프롬프트와 같은 언어로 작성."
    )
    selected_ids: list[str] = Field(
        description="후보 리스트에서 선택한 장소의 id. 동선 순서대로 5~6개. "
                    "반드시 후보 리스트의 id와 정확히 일치해야 함."
    )


# ============================================================
# LLM 셋업
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)
structured_llm = llm.with_structured_output(CourseResponse)

system_prompt = """[ABSOLUTE RULES — Never violate / 절대 규칙 — 위반 금지]
- This service ONLY recommends places in Seoul, South Korea.
  이 서비스는 오직 서울특별시 내의 장소만 추천한다.
- You MUST select places ONLY from the [Place Candidates] list below.
  반드시 [후보 장소 리스트]에서만 선택한다.
- Never invent place names, addresses, or IDs not in the candidate list.
  후보 리스트에 없는 장소를 만들어내지 않는다.
- Each id in selected_ids MUST exactly match an id from the candidate list.
  selected_ids의 각 id는 후보 리스트의 id와 문자 단위로 정확히 일치해야 한다.

[Language Rule / 언어 규칙]
- Respond in the SAME language as the user's request.
  사용자 요청 언어로 응답한다.
- Korean prompt → Korean description.
- English prompt → English description.

You are a knowledgeable local Seoul guide for travelers, especially foreign visitors.
너는 서울의 로컬 명소를 잘 아는 가이드로, 외국인 여행자를 포함한 손님을 위한 코스를 기획한다.

[Customer Info / 고객 정보]
- Schedule / 일정: {date}
- Preferred categories / 선호 테마: {categories}
- Group / 인원: {members}
- Budget / 예산: {budget}
- Request / 요청: {prompt}

[Time Estimation / 시간 추정 가이드라인]                       
Each place typically takes:
- Restaurant / 식당: 1.5h
- Cafe / 카페: 1h
- Cultural/Attraction / 문화·관광: 1.5h
- Shopping / 쇼핑: 1h
- Night view / 야경명소: 1h (must be after 17:00 / 17시 이후)

A 5~6 place course = 5~7 hours total.
Calculate: start_time + cumulative_durations = end_time

If end_time crosses 17:00 AND any nightview-* exists in candidates:
  → Include 1 nightview-* at the LAST or 2nd-LAST position.
If user explicitly says "morning only / 오전만 / 낮만":
  → Do NOT include nightview-* even if available.

[Course Order / 코스 순서 패턴]                                 
- 11~14시 시작: 식사 → 카페 → 명소 → 명소
- 14~17시 시작: 카페 → 명소 → 쇼핑 → 저녁식사 (or 야경)
- 17시+ 시작: 저녁식사 → 야경 → 카페/바
- AVOID consecutive same category (no restaurant→restaurant).
- Mix variety: at least 2 different prefixes from store-/culture-/tour-/nightview-.


[Congestion Data / 혼잡도 데이터]
{congestion_info}

[Place Candidates / 후보 장소 리스트] — Total {candidate_count} places
{candidates_text}

[Instructions / 지시사항]
1. Select 5~6 places from the candidate list above, in route order (geographically efficient).
   위 후보 중 5~6개를 동선이 효율적인 순서로 선택한다.
2. Mix variety: food (store-*), cultural facilities (culture-*), attractions etc.
   식당, 문화시설, 관광지를 균형있게 섞는다.
3. If user mentions "quiet/peaceful/한적한/조용한", avoid places marked as Crowded/붐빔.
4. Write 1~2 sentences of compelling description (in user's language).
5. Return ONLY description and selected_ids. Do not include other fields.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{language_instruction}\n\n위 조건에 맞춰 코스를 짜줘. / Plan a course based on the above.")
])


# ============================================================
# 헬퍼 함수
# ============================================================

def detect_response_language(prompt: str) -> str:
    """프롬프트 언어 감지. 한글 비율로 판단."""
    if not prompt:
        return 'ko'
    korean_chars = sum(1 for c in prompt if '\uac00' <= c <= '\ud7af')
    total_alpha = sum(1 for c in prompt if c.isalpha())
    if total_alpha == 0:
        return 'ko'
    ratio = korean_chars / total_alpha
    return 'ko' if ratio > 0.3 else 'en'

def format_candidates(candidates: list[PlaceCandidate]) -> str:
    """후보 리스트를 LLM이 읽기 좋은 텍스트로 변환"""
    if not candidates:
        return "(no candidates / 후보 없음)"
    
    lines = []
    for c in candidates:
        # introduction을 80자로 잘라 토큰 절약
        intro = ""
        if c.introduction:
            intro_clean = c.introduction.strip()
            intro = f" | {intro_clean[:80]}"
        
        lines.append(
            f"- id={c.id} | name={c.name} | category={c.category} "
            f"| address={c.address}{intro}"
        )
    return "\n".join(lines)


def format_congestion(congestion_data: list[SpotCongestion]) -> str:
    """혼잡도 데이터를 텍스트로 변환"""
    if not congestion_data:
        return "(no congestion data / 혼잡도 데이터 없음)"
    
    valid = [
        f"- {c.areaNm}: {c.congestionLabel}"
        for c in congestion_data
        if c.areaNm and c.congestionLabel and c.congestionLabel.strip() != "정보 없음"
    ]
    return "\n".join(valid) if valid else "(no valid congestion data)"


def validate_selected_ids(selected_ids: list[str], candidates: list[PlaceCandidate]) -> list[str]:
    """LLM이 헛소리한 ID를 걸러냄. 후보에 없는 ID는 제외."""
    candidate_ids = {c.id for c in candidates}
    valid = []
    invalid = []
    for sid in selected_ids:
        if sid in candidate_ids:
            valid.append(sid)
        else:
            invalid.append(sid)
    if invalid:
        logger.warning(f"LLM이 후보에 없는 ID를 반환함, 필터링: {invalid}")
    return valid


# ============================================================
# 엔드포인트
# ============================================================

@app.post("/api/ai/course", response_model=CourseResponse)
async def generate_course(request: CourseRequest):
    logger.info(
        f"AI 코스 생성 요청: 카테고리={request.categories}, "
        f"프롬프트='{request.prompt}', 후보 {len(request.candidates)}개"
    )

    # 후보가 너무 적으면 일찍 실패
    if len(request.candidates) < 5:
        logger.error(f"후보 부족: {len(request.candidates)}개. 최소 5개 필요")
        raise HTTPException(
            status_code=422,
            detail=f"후보 장소가 부족합니다 ({len(request.candidates)}개)"
        )
    

    #  언어 감지
    detected_lang = detect_response_language(request.prompt)
    if detected_lang == 'en':
        language_instruction = (
        "🚨 CRITICAL LANGUAGE RULE: The user wrote in English. "
        "You MUST write the 'description' field in ENGLISH ONLY.\n\n"
        "📝 PLACE NAME FORMAT: When mentioning Korean place names in the description, "
        "use BILINGUAL format: 'English Name(한국어이름)'.\n"
        "Examples:\n"
        "  - 'Bukchon Hanok Village(북촌한옥마을)'\n"
        "  - 'Gyeongbokgung Palace(경복궁)'\n"
        "  - 'N Seoul Tower(N서울타워)'\n"
        "  - 'Insadong(인사동)'\n"
        "This dual notation helps foreign visitors navigate Korea — they can show "
        "the Korean name to locals or taxi drivers.\n\n"
        "⚠️ This rule applies to the 'description' text ONLY. "
        "The 'selected_ids' MUST remain exactly as the candidate IDs (e.g., 'culture-3'). "
        "Do not modify or translate the IDs."
    )
    else:
        language_instruction = (
            "🚨 언어 규칙: 사용자 요청은 한국어이다. "
            "description 필드는 반드시 한국어로 작성한다."
        )
    
    logger.info(f"[Language] 감지된 응답 언어: {detected_lang}")


    candidates_text = format_candidates(request.candidates)
    congestion_info = format_congestion(request.congestionData)

    logger.info(f"[Prompt] 후보 {len(request.candidates)}개 LLM에 전달")

    chain = prompt_template | structured_llm

    try:
        result: CourseResponse = chain.invoke({
            "date": request.date,
            "categories": ", ".join(request.categories),
            "members": request.members,
            "budget": request.budget,
            "prompt": request.prompt,
            "candidates_text": candidates_text,
            "candidate_count": len(request.candidates),
            "congestion_info": congestion_info,
            "language_instruction": language_instruction,
        })

        # ID 검증: LLM이 환각으로 만든 ID 제거
        validated_ids = validate_selected_ids(result.selected_ids, request.candidates)

        if len(validated_ids) < 4:
            logger.error(
                f"유효 ID 부족: LLM 반환 {len(result.selected_ids)}개, "
                f"검증 통과 {len(validated_ids)}개"
            )
            raise HTTPException(
                status_code=422,
                detail="AI가 충분한 장소를 선택하지 못했습니다. 다시 시도해주세요."
            )

        # 검증된 ID로 응답 재구성
        final_response = CourseResponse(
            description=result.description,
            selected_ids=validated_ids
        )

        logger.info(
            f"AI 코스 생성 완료: 선택 ID {len(validated_ids)}개 "
            f"(원본 {len(result.selected_ids)}개)"
        )
        logger.info(f"[Selected IDs] {validated_ids}")
        return final_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM 체인 실행 중 에러 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI 코스 생성에 실패했습니다.")