import os
from dotenv import load_dotenv

# .env 파일 안의 내용들을 파이썬 환경변수로 싹 불러옵니다.
load_dotenv()

# 불러온 키를 변수에 예쁘게 담아줍니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 만약 키가 없으면 서버 켜질 때 바로 에러를 뱉게 해서 실수를 방지합니다!
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다!")