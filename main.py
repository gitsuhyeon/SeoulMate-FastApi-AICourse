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


# --- 1. 안드로이드(Spring)에서 넘어오는 데이터 모양 (요청) ---
class CourseRequest(BaseModel):
    date: str              # 예: "월 오후 6-7시"
    categories: list[str]  # 예: ["#한식", "#관광"]
    members: str           # 예: 최소인원- 최대인원
    budget: str            # 예: 100000원
    prompt: str            # 예: "외국인 친구랑 갈 종로 투어 짜줘"

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
- 선호하는 테마(카테고리): {categories}

[고객의 특별 요청사항]
{prompt}

위 조건들을 완벽하게 반영해서, 코스에 대한 '매력적인 소개글(description)'과 '장소 목록(places)'을 작성해줘."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "위 조건에 맞춰서 코스를 짜줘!")
])

# --- 4. API 엔드포인트 오픈 ---
@app.post("/api/ai/course")
async def generate_course(request: CourseRequest):
    # 로거를 사용하면 알아서 [abc-123...] Trace ID가 찍힙니다!
    logger.info(f"AI 코스 생성 요청 수신: 카테고리={request.categories}, 프롬프트='{request.prompt}'")
    # LangChain 체인 연결 (프롬프트 -> 구조화된 LLM)
    chain = prompt_template | structured_llm
    
    # 안드로이드에서 받은 데이터를 프롬프트의 빈칸({date}, {categories} 등)에 쏙쏙 끼워 넣습니다.
    result = chain.invoke({
        "date": request.date,
        "categories": ", ".join(request.categories),
        "prompt": request.prompt
    })
    
    logger.info("AI 코스 생성 완료 및 응답 반환")
    # AI가 뱉어낸 결과를 그대로 JSON으로 반환!
    return result