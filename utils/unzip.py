import py7zr
from pathlib import Path
from datetime import datetime

low_path = Path("data/user_logs.csv.7z")
save_path = Path("data/")

print(f"[{datetime.now()}] 압축 파일 열기 시작")

with py7zr.SevenZipFile(low_path, mode='r') as z:
    print("압축 파일 확인 완료")
    
    file_names = z.getnames()
    print("압축 내부 파일:", file_names)

    z.extractall(path=save_path)

print(f"[{datetime.now()}] 압축 해제 완료")