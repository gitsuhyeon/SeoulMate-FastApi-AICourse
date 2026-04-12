from fastapi import FastAPI
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from logger import logger
from middleware import trace_id_middleware
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

class MeetupCongestionDto(BaseModel):
    level: str | None = None # 붐빔 
    label: str | None = None
    sourceType: str | None = None
    sourceMessage: str | None = None
    basisPlaceName: str | None = None
    ppltnMin: int | None = None
    ppltnMax: int | None = None
    updatedAt: str | None = None

# --- 1. 안드로이드(Spring)에서 넘어오는 데이터 모양 (요청) ---
class CourseRequest(BaseModel):
    date: str              # 예: "월 오후 6-7시"
    categories: list[str]  # 예: ["#한식", "#관광"]
    members: str           # 예: 최소인원- 최대인원
    budget: str            # 예: 100000원
    prompt: str            # 예: "외국인 친구랑 갈 종로 투어 짜줘"
    # 스프링 부트가 넘겨주는 착한가격업소 리스트 (없으면 빈 리스트)
    recommendedStores: list[GoodPriceStore] = []
    congestionData: list[MeetupCongestionDto] = [] 

# --- 2. AI가 무조건 지켜야 하는 완벽한 JSON 구조 (응답) ---
class Place(BaseModel):
    name: str = Field(description="장소의 이름 (예: 창덕궁)")
    lat: float = Field(description="장소의 위도 (예: 37.5814)")
    lng: float = Field(description="장소의 경도 (예: 126.9910)")

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
1. '필수 참고 데이터'에 식당이나 장소가 제공되었다면, 고객의 동선에 맞춰 우선적으로 코스에 포함시켜줘. (가게 이름과 주소를 정확히 사용할 것)
2. 제공된 데이터가 없거나(비어있거나) 코스를 구성하기에 부족하다면, 네가 알고 있는 서울의 유명 명소, 관광지, 카페 맛집을 자유롭게 섞어서 완벽한 코스를 만들어줘.
3. **반드시 최소 3개 이상의 장소로 이루어진 꽉 찬 풀(Full) 코스**를 완성
4. **'장소 혼잡도 데이터'가 존재하고, 사용자 프롬프트에 '한적한', '조용한', '평화로운' 등의 비슷한 단어가 온다면, 반드시 현재 '붐빔' 상태인 곳은 가급적 피하고 쾌적한 동선을 제안해줘.**
5. 무조건 정해진 JSON 형식으로 응답해.
위 조건들을 완벽하게 반영해서, 코스에 대한 '매력적인 소개글(description)'과 '장소 목록(places)'을 작성해줘."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "위 조건에 맞춰서 코스를 짜줘!")
])

# --- 4. API 엔드포인트 오픈 ---
@app.post("/api/ai/course")
async def generate_course(request: CourseRequest):
    # 로거를 사용하면 알아서 [abc-123...] Trace ID가 찍힙니다!
    logger.info(f"AI 코스 생성 요청 수신: 카테고리={request.categories}, 프롬프트='{request.prompt}', 식당 데이터={len(request.recommendedStores)}개 전달됨")
    
    # 스프링 부트가 보내준 [착한가게 데이터]를 AI가 읽기 편한 텍스트로 변환
    if request.recommendedStores:
        stores_info = "\n".join([
            f"- 이름: {s.shName} (업종: {s.indutyCodeSeName}) / 주소: {s.shAddr} / 정보: {s.shInfo}" 
            for s in request.recommendedStores
        ])
    else:
        stores_info = "현재 조건에 맞는 추천 착한가격업소가 없습니다. 자유롭게 서울 명소 4~5곳을 추천해주세요."
    
    # [혼잡도 데이터 파싱] 스프링에서 보낸 basisPlaceName과 label(한글) 사용
    if request.congestionData:
        # basisPlaceName이 존재하고, label이 '정보 없음'이 아닌 경우만 필터링
        valid_congestion = [
            f"- 장소: {c.basisPlaceName} / 예상 혼잡도: {c.label}"
            for c in request.congestionData 
            if c.basisPlaceName and c.label and c.label != "정보 없음"
        ]
        
        if valid_congestion:
            congestion_info = "\n".join(valid_congestion)
        else:
            congestion_info = "현재 유효한 혼잡도 데이터가 없습니다."
    else:
        congestion_info = "현재 제공된 혼잡도 데이터가 없습니다."

    # LangChain 체인 연결 (프롬프트 -> 구조화된 LLM)
    chain = prompt_template | structured_llm
    
    # 안드로이드에서 받은 데이터를 프롬프트의 빈칸({date}, {categories} 등)에 쏙쏙 끼워 넣습니다.
    result = chain.invoke({
        "date": request.date,
        "categories": ", ".join(request.categories),
        "members": request.members,
        "budget": request.budget,
        "prompt": request.prompt,
        "stores_info": stores_info,  # 변환된 가게 정보 주입!
        "congestion_info": congestion_info
    })
    
    logger.info("AI 코스 생성 완료: 총 {len(result.places)}개 장소 반환")
    # AI가 뱉어낸 결과를 그대로 JSON으로 반환!
    return result