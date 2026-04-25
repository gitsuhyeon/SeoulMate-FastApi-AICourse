# security.py
import os
import logging
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# 실서버 배포 시 Render 환경변수에 FASTAPI_SECRET_KEY를 등록하세요.
FASTAPI_SECRET_KEY = os.getenv("AI_SERVER_SECRET")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key를 검증하는 전역 문지기 함수"""
    #logger.warning(f"수신된 API KEY: '{api_key}', 길이: {len(api_key)}")#문제 없으면 삭제해
    if api_key == FASTAPI_SECRET_KEY:
        return api_key
    
    logger.warning("🚨 잘못된 API Key로 접근 시도가 있었습니다!")
    raise HTTPException(status_code=403, detail="접근 권한이 없습니다 (Invalid API Key)")