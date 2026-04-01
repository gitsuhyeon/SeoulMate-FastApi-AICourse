import uuid
from fastapi import Request
from logger import trace_id_context

async def trace_id_middleware(request: Request, call_next):
    # 1. Spring Boot에서 넘겨준 X-Trace-Id를 헤더에서 찾습니다. (없으면 새로 생성)
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    
    # 2. 로거가 볼 수 있도록 상자에 담습니다.
    token = trace_id_context.set(trace_id)
    
    try:
        # 3. 실제 API(main.py) 로직을 실행하러 보냅니다.
        response = await call_next(request)
        
        # (보너스) 응답 헤더에도 이 ID를 넣어서 돌려주면 디버깅이 훨씬 편해집니다.
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        # 4. 일이 끝나면 상자를 비워줍니다. (메모리 누수 방지)
        trace_id_context.reset(token)