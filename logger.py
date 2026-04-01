import logging
import os
from contextvars import ContextVar

# 1. Trace ID를 담아둘 마법의 상자 (전역 변수처럼 쓰지만 쓰레드/비동기에 안전함)
trace_id_context: ContextVar[str] = ContextVar("trace_id", default="NoTrace")

# 2. 로그가 찍힐 때마다 상자에서 ID를 꺼내주는 필터
class TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = trace_id_context.get()
        return True

def get_logger():
    # 3. AWS 배포를 고려한 상대 경로 및 환경변수 사용
    log_dir = os.getenv("LOG_DIR", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "fastapi.log")

    logger = logging.getLogger("seoul_link_ai")
    logger.setLevel(logging.INFO)

    # 핸들러가 중복으로 추가되는 것 방지
    if not logger.handlers:
        # 포맷에 [%(trace_id)s] 추가!
        formatter = logging.Formatter('%(asctime)s [%(trace_id)s] [%(levelname)s] %(name)s - %(message)s')

        # 파일 출력 (Promtail용)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # 콘솔 출력 (개발자 터미널용)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.addFilter(TraceIdFilter())

    return logger

# 다른 파일에서 이 logger를 import 해서 사용합니다.
logger = get_logger()