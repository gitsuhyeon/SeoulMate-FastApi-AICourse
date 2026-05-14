from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from math import radians, sin, cos, asin, sqrt
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from logger import logger
from middleware import trace_id_middleware
from security import verify_api_key

app = FastAPI(dependencies=[Depends(verify_api_key)])
# ---------------------------------------------------
# 로컬테스트용이라 테스트후 위에 것 살리기
# app = FastAPI()

app.middleware("http")(trace_id_middleware)

logger.info("FastAPI LangChain 서버가 성공적으로 시작되었습니다!")

# ============================================================
# Health check — Render warmup용 (GitHub Actions cron이 14분마다 호출)
# 글로벌 verify_api_key 의존성은 그대로 적용되므로 ping 측에서 X-API-Key 헤더 필요.
# ============================================================
 
@app.get("/health")
async def health_check():
    """Render free tier sleep 방지용 가벼운 endpoint."""
    return {"status": "ok", "service": "seoul-link-ai"}


# ============================================================
# 요청 모델 (Spring → FastAPI)
# ============================================================

class PlaceCandidate(BaseModel):
    """Spring이 보내는 후보 장소"""
    id: str = Field(description="후보 ID. 'store-XXX' / 'culture-XXX' / 'tour-XXX' / 'nightview-XXX'")
    name: str
    address: str
    category: str = Field(default="", description="한식, 미술관/갤러리, 공연장 등")
    introduction: str | None = Field(default=None, description="짧은 소개")
    # ★ 추가: Spring이 이미 보내고 있는 좌표를 받기 (통신 contract 변경 아님 — Pydantic이 무시하던 필드를 수신)
    lat: float | None = None
    lng: float | None = None


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
    candidates: list[PlaceCandidate] = []
    congestionData: list[SpotCongestion] = []


# ============================================================
# 응답 모델 (FastAPI → Spring) — 스키마 변경 금지
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

# temperature 0.7 → 0.5 (창의성보다 일관성)
# timeout / max_retries 명시: Render cold start나 OpenAI 일시적 장애 대응
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=OPENAI_API_KEY,
    timeout=60.0,
    max_retries=2,
)
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
- Select EXACTLY 5 or 6 places. Never fewer than 5.
  반드시 5개 또는 6개를 선택한다. 4개 이하는 절대 금지.
- Never repeat the same id twice in selected_ids.
  selected_ids 내에 같은 id를 두 번 넣지 않는다.

[Language Rule / 언어 규칙]
- Respond in the SAME language as the user's request.
  반드시 사용자 요청 언어로 응답한다.
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
1. Select EXACTLY 5~6 places from the candidate list above, in route order (geographically efficient).
   위 후보 중 정확히 5~6개를 동선이 효율적인 순서로 선택한다.
2. Mix variety: food (store-*), cultural facilities (culture-*), attractions etc.
   식당, 문화시설, 관광지를 균형있게 섞는다.
3. If user mentions "quiet/peaceful/한적한/조용한", avoid places marked as Crowded/붐빔.
4. Write 1~2 sentences of compelling description (in user's language).
   The description MUST clearly identify this as a Seoul day course.
   description은 명확히 '서울 일일 코스'임을 알 수 있게 작성한다.
   If the user mentioned an ambiguous region (e.g., "Korea", "한국"),
   explicitly include "Seoul" / "서울" in the description.
5. Return ONLY description and selected_ids. Do not include other fields.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{language_instruction}\n\n위 조건에 맞춰 코스를 짜줘. / Plan a course based on the above.")
])


# ============================================================
# 헬퍼 함수
# ============================================================

# 목표 코스 길이
TARGET_MIN = 5
TARGET_MAX = 6
HARD_MIN = 4  # 이 미만이면 진짜 실패로 간주


def detect_response_language(prompt: str) -> str:
    """
    한국어 vs 비한국어만 판별. 비한국어의 구체적인 언어(영어/일본어/중국어 등)는
    LLM(멀티링구얼)이 자체적으로 감지·응답하도록 위임.
    """
    if not prompt:
        return 'ko'
    korean_chars = sum(1 for c in prompt if '\uac00' <= c <= '\ud7af')
    total_alpha = sum(1 for c in prompt if c.isalpha())
    if total_alpha == 0:
        return 'ko'
    ratio = korean_chars / total_alpha
    return 'ko' if ratio > 0.3 else 'non_ko'


def format_candidates(candidates: list[PlaceCandidate]) -> str:
    """후보 리스트를 LLM이 읽기 좋은 텍스트로 변환"""
    if not candidates:
        return "(no candidates / 후보 없음)"

    lines = []
    for c in candidates:
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


def dedupe_preserve_order(ids: list[str]) -> list[str]:
    """순서 유지하면서 중복 제거."""
    seen = set()
    result = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def validate_selected_ids(selected_ids: list[str], candidates: list[PlaceCandidate]) -> list[str]:
    """LLM이 환각으로 만든 ID를 걸러냄. 후보에 없는 ID는 제외."""
    candidate_ids = {c.id for c in candidates}
    valid, invalid = [], []
    for sid in selected_ids:
        if sid in candidate_ids:
            valid.append(sid)
        else:
            invalid.append(sid)
    if invalid:
        logger.warning(f"LLM이 후보에 없는 ID를 반환함, 필터링: {invalid}")
    return valid


def repair_selected_ids(
    valid_ids: list[str],
    candidates: list[PlaceCandidate],
    target: int = TARGET_MIN,
) -> list[str]:
    """
    valid_ids가 target개 미만이면 후보에서 보충. 다양성 우선.
    - 이미 선택된 prefix와 다른 prefix 후보 먼저
    - 그 다음 같은 prefix 후보
    - 야경(nightview-*)은 보충에서 제외 — 시간대 조건이 없으면 끼면 안 됨
    """
    if len(valid_ids) >= target:
        return valid_ids

    used = set(valid_ids)
    used_prefixes = {sid.split('-')[0] for sid in valid_ids}

    pool = [c for c in candidates if c.id not in used and not c.id.startswith("nightview-")]
    diff_prefix = [c for c in pool if c.id.split('-')[0] not in used_prefixes]
    same_prefix = [c for c in pool if c.id.split('-')[0] in used_prefixes]
    fill_order = diff_prefix + same_prefix

    needed = target - len(valid_ids)
    repaired = list(valid_ids)
    for c in fill_order[:needed]:
        repaired.append(c.id)
        logger.info(f"[Repair] 부족분 보충: {c.id} ({c.name})")
    return repaired


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 점 사이 거리(미터)."""
    R = 6_371_000.0
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * R * asin(sqrt(a))


def optimize_route_nn(
    ids: list[str],
    candidate_map: dict[str, PlaceCandidate],
) -> list[str]:
    """
    Nearest-neighbor 정렬.
    - 시작점은 LLM이 뽑은 첫 번째 ID로 고정 (LLM이 시간대/카테고리 패턴 알고 있음)
    - 그 다음부터 직전 장소와 가장 가까운 순서로 재배열
    - 좌표 누락 ID가 하나라도 있으면 정렬 포기하고 LLM 순서 유지
    """
    if len(ids) <= 2:
        return ids

    places = []
    for sid in ids:
        c = candidate_map.get(sid)
        if c is None or c.lat is None or c.lng is None:
            logger.info(f"[NN] 좌표 누락으로 정렬 skip: {sid}")
            return ids
        places.append(c)

    ordered = [places[0]]
    remaining = places[1:]
    while remaining:
        last = ordered[-1]
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: haversine_m(last.lat, last.lng, remaining[i].lat, remaining[i].lng),
        )
        ordered.append(remaining.pop(nearest_idx))

    new_ids = [p.id for p in ordered]
    if new_ids != ids:
        logger.info(f"[NN] 동선 재정렬: {ids} → {new_ids}")
    return new_ids


def push_nightview_to_end(ids: list[str]) -> list[str]:
    """야경명소는 항상 마지막. 시스템 프롬프트의 LAST 지시를 후처리로 보장."""
    nightviews = [i for i in ids if i.startswith("nightview-")]
    others = [i for i in ids if not i.startswith("nightview-")]
    if nightviews and others and ids != others + nightviews:
        logger.info(f"[Post] 야경명소를 마지막으로 이동: {ids} → {others + nightviews}")
    return others + nightviews


def cap_to_max(ids: list[str], max_len: int = TARGET_MAX) -> list[str]:
    """6개 초과 시 잘라냄. LLM이 7개 이상 뽑는 경우 방어."""
    if len(ids) > max_len:
        logger.info(f"[Cap] {len(ids)}개 → {max_len}개로 자름")
        return ids[:max_len]
    return ids


# ============================================================
# 엔드포인트
# ============================================================

@app.post("/api/ai/course", response_model=CourseResponse)
async def generate_course(request: CourseRequest):
    logger.info(
        f"AI 코스 생성 요청: 카테고리={request.categories}, "
        f"프롬프트='{request.prompt}', 후보 {len(request.candidates)}개"
    )

    # 후보가 너무 적으면 일찍 실패 — 보충해도 의미 없음
    if len(request.candidates) < TARGET_MIN:
        logger.error(f"후보 부족: {len(request.candidates)}개. 최소 {TARGET_MIN}개 필요")
        raise HTTPException(
            status_code=422,
            detail=f"후보 장소가 부족합니다 ({len(request.candidates)}개)"
        )

    # 언어 감지
    detected_lang = detect_response_language(request.prompt)
    if detected_lang == 'non_ko':
        language_instruction = (
            "🚨 CRITICAL LANGUAGE RULE: The user did NOT write in Korean. "
            "Detect their actual language (English, Japanese, Chinese, Spanish, etc.) "
            "and write the 'description' field in THAT SAME language.\n"
            "If multiple languages are mixed, use the dominant non-Korean one.\n\n"
            "📝 PLACE NAME FORMAT: When mentioning Korean place names in the description, "
            "ALWAYS include the original Korean in parentheses for navigation purposes.\n"
            "Use a transliteration or local-language name appropriate to the user's language, "
            "followed by '(한국어원문)'.\n"
            "Examples:\n"
            "  - English user → 'Bukchon Hanok Village(북촌한옥마을)'\n"
            "  - Japanese user → '北村韓屋村(북촌한옥마을)'\n"
            "  - Chinese user → '北村韩屋村(북촌한옥마을)'\n"
            "  - Spanish user → 'Aldea Hanok de Bukchon(북촌한옥마을)'\n"
            "  - Any other language → 'LocalName(북촌한옥마을)'\n\n"
            "This dual notation helps foreign visitors show the original Korean name to "
            "taxi drivers or Korean map apps — which works regardless of their language.\n\n"
            "⚠️ This rule applies to the 'description' text ONLY. "
            "The 'selected_ids' MUST remain exactly as the candidate IDs "
            "(e.g., 'culture-3', 'tour-126508'). Do not modify or translate the IDs."
        )
    else:
        language_instruction = (
            "🚨 언어 규칙: 사용자 요청은 한국어이다. "
            "description 필드는 반드시 한국어로 작성한다."
        )

    logger.info(f"[Language] 감지된 응답 언어 그룹: {detected_lang}")

    candidates_text = format_candidates(request.candidates)
    congestion_info = format_congestion(request.congestionData)
    candidate_map = {c.id: c for c in request.candidates}

    chain = prompt_template | structured_llm

    invoke_payload = {
        "date": request.date,
        "categories": ", ".join(request.categories),
        "members": request.members,
        "budget": request.budget,
        "prompt": request.prompt,
        "candidates_text": candidates_text,
        "candidate_count": len(request.candidates),
        "congestion_info": congestion_info,
        "language_instruction": language_instruction,
    }

    # ============================================================
    # LLM 호출: 결과가 부실하면 1회 재시도
    # ============================================================
    MAX_ATTEMPTS = 2
    result: CourseResponse | None = None
    validated_ids: list[str] = []
    last_exception: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info(f"[LLM] 호출 시도 {attempt}/{MAX_ATTEMPTS}")
            # async endpoint 안에서는 ainvoke + await가 필수.
            # invoke()를 쓰면 LLM 응답(수 초) 동안 이벤트 루프가 블록되어
            # 동시 요청이 처리되지 않음.
            result = await chain.ainvoke(invoke_payload)

            # 환각 ID 제거 → 중복 제거
            validated_ids = validate_selected_ids(result.selected_ids, request.candidates)
            validated_ids = dedupe_preserve_order(validated_ids)

            logger.info(
                f"[LLM] 시도 {attempt} 결과: LLM 반환 {len(result.selected_ids)}개 → "
                f"환각/중복 필터 후 {len(validated_ids)}개"
            )

            if len(validated_ids) >= TARGET_MIN:
                break  # 충분 — 보충 불필요

            if attempt < MAX_ATTEMPTS:
                logger.warning(f"[LLM] 결과 부실 (유효 {len(validated_ids)}개), 재시도")

        except Exception as e:
            last_exception = e
            logger.warning(f"[LLM] 시도 {attempt} 예외: {e}")
            if attempt >= MAX_ATTEMPTS:
                break

    # 두 시도 모두 LLM 자체 실패
    if result is None:
        logger.error(f"LLM 호출 전체 실패: {last_exception}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI 코스 생성에 실패했습니다.")

    # ============================================================
    # 결과 안정화: 보충 → 동선 정렬 → 야경 후처리 → 길이 제한
    # ============================================================

    # 1) 부족분 보충 (시연 안정성 — 422 던지지 않고 후보로 채움)
    if len(validated_ids) < TARGET_MIN:
        if len(validated_ids) < HARD_MIN:
            logger.error(
                f"유효 ID 심각 부족: {len(validated_ids)}개 (HARD_MIN={HARD_MIN}). "
                f"후보로 보충 시도"
            )
        validated_ids = repair_selected_ids(validated_ids, request.candidates, target=TARGET_MIN)

    # 보충해도 부족하면 진짜 실패
    if len(validated_ids) < HARD_MIN:
        logger.error(f"보충 후에도 유효 ID 부족: {len(validated_ids)}개")
        raise HTTPException(
            status_code=422,
            detail="AI가 충분한 장소를 선택하지 못했습니다. 다시 시도해주세요."
        )

    # 2) 6개 초과 자르기
    validated_ids = cap_to_max(validated_ids, max_len=TARGET_MAX)

    # 3) Nearest-neighbor로 동선 재정렬
    optimized_ids = optimize_route_nn(validated_ids, candidate_map)

    # 4) 야경은 마지막으로 강제 이동
    final_ids = push_nightview_to_end(optimized_ids)

    final_response = CourseResponse(
        description=result.description,
        selected_ids=final_ids,
    )

    logger.info(
        f"AI 코스 생성 완료: 최종 {len(final_ids)}개 (원본 LLM {len(result.selected_ids)}개)"
    )
    logger.info(f"[Final IDs] {final_ids}")
    # description도 함께 로그 (multi-line → 한 줄로 정리해서 Loki/Render 검색 편의)
    # 만약 로그 용량 커지면 삭제
    desc_oneline = " ".join(result.description.split()) if result.description else "(empty)"
    logger.info(f"[Final Description] {desc_oneline}")
    return final_response