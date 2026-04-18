from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from logger import logger
from middleware import trace_id_middleware
from fastapi import HTTPException
from typing import Optional
import logging
#vscode라 가상환경 못잡을때 cmd+shift+p > Python: Select Interpreter > 가상환경체크

app = FastAPI()
# 미들웨어 장착
app.middleware("http")(trace_id_middleware)

logger.info("FastAPI LangGraph 서버가 성공적으로 시작되었습니다!")

# 스프링 부트의 GoodPriceStore 엔티티 모양 그대로 받기 (CamelCase 일치)
class GoodPriceStore(BaseModel):
    shId: str | None = None
    shName: str | None = None
    indutyCodeSeName: str | None = None
    shAddr: str | None = None
    shInfo: str | None = None
    lat: float | None = None 
    lng: float | None = None

class SpotCongestion(BaseModel):
    areaNm: str | None = None
    congestionLevel: str | None = None
    congestionLabel: str | None = None
    ppltnMin: int | None = None
    ppltnMax: int | None = None
    observedAt: str | None = None

# --- 1. 안드로이드(Spring)에서 넘어오는 데이터 모양 (요청) ---
class CourseRequest(BaseModel):
    date: str              # 예: "월 오후 6-7시"
    categories: list[str]  # 예: ["#한식", "#관광"]
    members: str           # 예: 최소인원- 최대인원
    budget: str            # 예: 100000원
    prompt: str            # 예: "외국인 친구랑 갈 종로 투어 짜줘"
    # 스프링 부트가 넘겨주는 착한가격업소 리스트 (없으면 빈 리스트)
    recommendedStores: list[GoodPriceStore] = []
    congestionData: list[SpotCongestion] = []
# --- 2. AI가 무조건 지켜야 하는 완벽한 JSON 구조 (응답) ---
class Place(BaseModel):
    name: str = Field(description="장소의 이름 (예: 창덕궁)")
    address: str = Field(description="해당 장소의 도로명/지번 주소. 정확히 모르면 대략적인 지역명(예: 서울 종로구)만 작성할 것")

class CourseResponse(BaseModel):
    description: str = Field(description="이 코스에 대한 1~2줄의 매력적인 소개글 (기획 의도)")
    places: list[Place] = Field(description="추천 장소 리스트 (동선 순서대로 배열)")

# --- 3. LLM 및 프롬프트 셋팅 ---
# 실행 시 터미널 환경변수에 OPENAI_API_KEY가 등록되어 있어야 합니다.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)

# 핵심 마법: AI에게 "무조건 CourseResponse 모양(JSON)으로만 대답해!" 라고 강제합니다.
structured_llm = llm.with_structured_output(CourseResponse)

system_prompt = """너는 서울의 숨겨진 로컬 명소를 기가 막히게 잘 아는 현지인 가이드야.
외국인 여행자와 함께할 투어 코스를 기획해야 해.

[고객의 상황 및 취향]
- 투어 일정: {date}
- 선호하는 테마: {categories}
- 참여 인원: {members}
- 예산: {budget}

[고객의 특별 요청사항]
{prompt}

[필수 참고 데이터: 서울시 착한 가격업소 추천 리스트]
{stores_info}

[선택 참고 데이터: 주요 장소 혼잡도 데이터]
{congestion_info}

[지시사항]
1. '필수 참고 데이터'에 식당이나 장소가 제공되었다면, 고객의 동선에 맞춰 우선적으로 1개만 코스에 포함시켜줘. (가게 이름과 주소를 정확히 사용할 것)
2. 제공된 데이터가 없거나(비어있거나) 코스를 구성하기에 부족하다면, 네가 알고 있는 서울의 유명 명소, 관광지, 카페 (1-2개만) 맛집(1개만)을 자유롭게 섞어서 완벽한 코스를 만들어줘.
3. **🚨 절대 가짜 위도/경도(GPS) 좌표를 만들지 마. 반드시 해당 장소의 정확한 'address(주소)'만 정확한 텍스트로 제공해.**
4. **반드시 최소 5개 이상의 장소로 이루어진 꽉 찬 풀(Full) 코스**를 완성
5. **'장소 혼잡도 데이터'가 존재하고, 사용자 프롬프트에 '한적한', '조용한', '평화로운' 등의 비슷한 단어가 온다면, 반드시 현재 '붐빔' 상태인 곳은 가급적 피하고 쾌적한 동선을 제안해줘.**
6. 무조건 정해진 JSON 형식으로 응답해.
위 조건들을 완벽하게 반영해서, 코스에 대한 '매력적인 소개글(description)'과 '장소 목록(places)'을 작성해줘."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "위 조건에 맞춰서 코스를 짜줘!")
])

logger = logging.getLogger(__name__)

# --- 4. API 엔드포인트 오픈 ---
@app.post("/api/ai/course")
async def generate_course(
    request: CourseRequest,
    x_trace_id: Optional[str] = Header(default="NoTrace", alias="X-Trace-Id")
    ):
    # 로거를 사용하면 알아서 [abc-123...] Trace ID가 찍힙니다!
    logger.info(f"[{x_trace_id}] AI 코스 생성 요청 수신: 카테고리={request.categories}, 프롬프트='{request.prompt}', 식당 데이터={len(request.recommendedStores)}개 전달됨")
    

    #  스프링 부트가 보내준 가게 정보 + [위도/경도]를 묶어서 AI에게 전달
    if request.recommendedStores:
        stores_list = []
        for s in request.recommendedStores:
            # null 처리: 좌표가 있으면 프롬프트에 포함, 없으면 생략
            coord_str = f" (위도: {s.lat}, 경도: {s.lng})" if s.lat and s.lng else ""
            stores_list.append(f"- 이름: {s.shName} (업종: {s.indutyCodeSeName}) / 정보: {s.shInfo} / 주소: {s.shAddr}{coord_str}")
        stores_info = "\n".join(stores_list)
    else:
        stores_info = "조건에 맞는 가게가 없습니다. 자유롭게 서울 명소 4~5곳을 추천해주세요."
    
    #  [혼잡도 데이터 파싱] 스프링에서 보낸 필드명(areaNm, congestionLabel)으로 수정
    if request.congestionData:
        valid_congestion = [
            f"- 장소: {c.areaNm} / 예상 혼잡도: {c.congestionLabel}"
            for c in request.congestionData 
            if c.areaNm and c.congestionLabel and c.congestionLabel.strip() != "정보 없음"
        ]
        
        if valid_congestion:
            congestion_info = "\n".join(valid_congestion)
        else:
            congestion_info = "현재 유효한 혼잡도 데이터가 없습니다."
    else:
        congestion_info = "현재 제공된 혼잡도 데이터가 없습니다."

    # LM에게 주입될 '혼잡도 데이터 원본' 로그 확인
    logger.info(f"[{x_trace_id}][Prompt Input] 전달된 혼잡도 정보:\n{congestion_info}")

    # LangChain 체인 연결 (프롬프트 -> 구조화된 LLM)
    chain = prompt_template | structured_llm

    try:
        result = chain.invoke({
            "date": request.date,
            "categories": ", ".join(request.categories),
            "members": request.members,
            "budget": request.budget,
            "prompt": request.prompt,
            "stores_info": stores_info,
            "congestion_info": congestion_info
        })
        logger.info(f"[{x_trace_id}]AI 코스 생성 완료: 총 {len(result.places)}개 장소 반환")
        return result
        
    except Exception as e:
        logger.error(f"[{x_trace_id}]LLM 체인 실행 중 에러 발생: {str(e)}")
        # FastAPI에서 클라이언트(스프링/안드로이드)에게 적절한 500 에러 응답을 내려주는 처리 필요
        raise HTTPException(status_code=500, detail="AI 코스 생성에 실패했습니다.")
