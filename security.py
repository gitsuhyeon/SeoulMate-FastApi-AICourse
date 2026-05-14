# security.py
import os
import logging
from fastapi import Security, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) 
# True에서 False로 바꾼 이유는, uptimerobot으로 render 깨우기위해 ping을 보내기 위함
# 이때, 보안을 위해 header 없이 보낼거라, header 없어도 함수 호출되도록

# Render 환경변수에 FASTAPI_SECRET_KEY를 등록해놓음
FASTAPI_SECRET_KEY = os.getenv("AI_SERVER_SECRET")

# 인증 면제 path (모니터링/헬스체크 등)
PUBLIC_PATHS = {"/health"}

async def verify_api_key(
        request: Request,
        api_key: str = Security(api_key_header)
    ):
    """API Key를 검증하는 전역 문지기 함수. 단, PUBLIC_PATHS는 면제."""
    if request.url.path in PUBLIC_PATHS:
        return None
    #logger.warning(f"수신된 API KEY: '{api_key}', 길이: {len(api_key)}")#문제 없으면 삭제해
    if api_key == FASTAPI_SECRET_KEY:
        return api_key
    
    logger.warning("🚨 잘못된 API Key로 접근 시도가 있었습니다: path={request.url.path}")
    raise HTTPException(status_code=403, detail="접근 권한이 없습니다 (Invalid API Key)")