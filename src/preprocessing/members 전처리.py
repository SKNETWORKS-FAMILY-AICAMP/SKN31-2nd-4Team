# - pandas는 CSV 파일을 읽고 데이터프레임으로 처리하기 위해 사용한다.
# - numpy는 결측값(np.nan), 숫자 계산 처리를 위해 사용한다.
import pandas as pd
import numpy as np

# - members_v3.csv를 pandas DataFrame으로 불러온다.
# - 이후 모든 전처리는 members라는 변수에 담긴 데이터프레임을 기준으로 진행한다.
members = pd.read_csv("members_v3.csv")

# 파일 경로 설정
members = pd.read_csv("C:\documents\SKN31-2nd-4Team/members_v3.csv")

print(members.shape) # 행과 개수 확인
print(members.head()) # 앞 5행을 확인해서 실제 데이터 모양으로 보기
print(members.info()) # 컬럼명, 데이터 타입, 결측치 여부 확인

# 4. 결측치 개수 확인

print(members.isnull().sum())

# - 어떤 컬럼에 결측치가 많은지 확인한다.

# - 실제 members_v3.csv에서는 gender 결측치가 매우 많다.
# - gender는 절반 이상 비어 있으므로 삭제보다는 unknown으로 처리하는 것이 안전하다.

members["gender"] = members["gender"].fillna("unknown")

# 5. 중복회원 확인
print(members["msno"].duplicated().sum())

# - msno는 회원 ID이므로 원칙적으로 한 명당 한 행이어야 한다.
# - 만약 중복이 있다면 merge할 때 행 수가 늘어날 수 있다.
# - members_v3.csv는 회원 기본 정보라 보통 msno 중복이 없어야 한다.

# 중복이 있을 경우 처리 코드:

members = members.drop_duplicates(subset="msno", keep="last")

# 6. bd 컬럼 확인 시 이상치 많아 드랍한다. 

members = members.drop(columns=["bd"])

# # registration_init_time 날짜 변환
# # ------------------------------------------------------------
# 기존 형태:
# - 20110911
# - 20150203
# - 20170331
members["registration_init_time"] = pd.to_datetime(
    members["registration_init_time"],
    format="%Y%m%d",
    errors="coerce"
)

# 이유:
# - 원본은 숫자처럼 보이지만 실제 의미는 날짜다.
# - 날짜형으로 바꿔야 연도, 월, 일, 요일을 추출할 수 있다.
# - errors="coerce"는 날짜 변환에 실패한 값을 NaT로 바꾼다.
# - 오류로 코드가 멈추는 것을 방지한다.


# 2. 연/월/일 생성
members["reg_year"] = members["registration_init_time"].dt.year
members["reg_month"] = members["registration_init_time"].dt.month
members["reg_day"] = members["registration_init_time"].dt.day

# 3. reg_time_num 생성
members["reg_time_num"] = (
    members["reg_year"]
    + (members["reg_month"] - 1) / 12
    + members["reg_day"] / 365
)

import datetime 
# 날짜 변환 후 결측치 처리
print(members["registration_init_time"].isnull().sum())

# 이유:
# - 날짜 변환에 실패한 값이 있는지 확인한다.
# - 만약 결측치가 생겼다면 원본 날짜 형식이 잘못된 행이 있다는 뜻이다.


# 날짜 파생변수 생성
members["reg_year"] = members["registration_init_time"].dt.year
members["reg_month"] = members["registration_init_time"].dt.month
members["reg_day"] = members["registration_init_time"].dt.day
members["reg_weekday"] = members["registration_init_time"].dt.weekday

# - 모델은 날짜 자체보다 연도, 월, 요일 같은 숫자 특징을 더 쉽게 사용한다.
# - 가입 연도: 오래된 회원인지 신규 회원인지 판단 가능
# - 가입 월: 특정 시기 가입자의 이탈 경향 확인 가능
# - 가입 요일: 가입 패턴 확인 가능

# ------------------------------------------------------------
# 15. 논문 방식의 연속형 날짜 변수 생성
# ------------------------------------------------------------

# 논문에서 사용한 방식:


members["reg_time_num"] = (
    members["reg_year"]
    + (members["reg_month"] - 1) / 12
    + members["reg_day"] / 365
)

# 이유:
# - 날짜를 하나의 연속적인 숫자로 표현하는 방식이다.
# - 예를 들어 2015년 1월보다 2016년 1월이 더 큰 숫자가 된다.
# - 모델이 가입 시점의 시간 흐름을 숫자로 이해할 수 있다.

# registered_via 값 확인 --> [4, 3, 9, 7]을 제외하고 나머지는 0으로 채우기

members["registered_via"] = members["registered_via"].where(
    members["registered_via"].isin([4, 3, 9, 7]),
    0
)

#  gender 더미화
members = pd.get_dummies(
    members,
    columns=["gender"],
    dummy_na=False
)

# - 머신러닝 모델은 male, female, unknown 같은 문자열을 그대로 이해하지 못한다.
# - 따라서 숫자형 0/1 컬럼으로 바꿔야 한다.
# 이미 11단계에서 결측치를 unknown으로 바꿨기 때문에 dummy_na=True를 쓸 필요는 없다.

print(members.info())

# - 문자열 컬럼이 남아 있는지 확인한다.
# - gender는 더미화했으므로 object 타입은 msno만 남는 것이 정상이다.
# - msno는 ID이므로 모델 학습에는 사용하지 않는다.

# 최종 결측치 확인
print(members.isnull().sum().sort_values(ascending=False).head(20))

# - 전처리 후 결측치가 남아 있는지 확인한다.
# - bd, gender, 날짜 파생변수 쪽에 결측치가 남지 않았는지 확인한다.

members.to_csv(
    "members_v3_preprocessed.csv",
    index=False
)


# - 전처리한 members 데이터를 따로 저장한다.
# - 이후 train.csv, transactions.csv, user_logs.csv와 merge할 때 사용한다.

