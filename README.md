# SKN31-2st-4Team

## 팀 및 팀원 소개

### 팀 명
- 영크크Box(kkbox)🎵

### 팀원
| 안영선 | 김봉남 | 유진영 | 김효민 | 김세희
| :---: | :---: | :---: | :---: | :---: |
| <a href="https://github.com/dksdudtjs94"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/bongrybong"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/ujneg18-source"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/hyomin0357"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/kimsahee0401271111-collab"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> |
|<img src="images\마틴.png" width="150" height="150"> | <img src="images\주훈.png" width="150" height="150"> | <img src="images\제임스.png" width="150" height="150"> | <img src="images\성현.png" width="150" height="150"> | <img src="images\건호.png" width="150" height="150"> |
| <b>PM</b>, | <b>부PM</b>, | ㅇㅇ | ㅇㅇ | ㅇㅇ|

---

## 프로젝트 개요

### 프로젝트명
- KKBOX 음악 스트리밍 서비스 고객 이탈 예측 모델 개발

### 프로젝트 소개
이번 프로젝트는 아시아 최대 음악 스트리밍 서비스인 KKBOX의 실제 구독자 데이터를 활용하여 고객 이탈(Churn) 여부를 예측하는 머신러닝 모델을 개발하는 프로젝트입니다.

사용자의 결제 이력, 멤버십 정보, 음악 청취 로그 데이터를 통합 분석하여 구독 만료 후 일정 기간 내 재구독하지 않을 가능성을 예측하고, 이탈 위험 고객을 조기에 식별할 수 있는 데이터 기반 의사결정 시스템을 구축하는 것을 목표로 합니다.

또한 모델 결과를 Streamlit 기반 대시보드로 시각화하여 데이터 분석 결과와 고객 이탈 예측 기능을 직관적으로 제공합니다.

### 프로젝트 배경
구독 기반 비즈니스에서는 기존 고객 유지가 신규 고객 확보보다 비용 효율성이 높기 때문에 고객 이탈 관리는 핵심 경영 과제로 평가됩니다. 특히 음악 스트리밍 서비스의 경우 구독자의 이탈률이 소폭만 증가하더라도 기업 매출에 직접적인 영향을 미칠 수 있습니다.

이번 프로젝트는 실제 기업인 KKBOX가 개최한 Kaggle 경진대회 데이터를 활용하여 진행되는데 해당 데이터는 실제 서비스 운영 과정에서 수집된 대규모 데이터로 현실적인 고객 행동 패턴과 다양한 변수들을 포함하고 있어 실무 환경과 유사한 분석 경험을 제공합니다.

또한 Kaggle 공식 평가 지표인 Log Loss를 활용하여 모델 성능을 객관적으로 평가할 수 있으며, 글로벌 참가자들과 성능을 비교할 수 있다는 점에서 프로젝트의 실용성과 의미가 큽니다.

### 프로젝트 목표
프로젝트의 주요 목표는 구독 만료 후 30일 이내 재구독하지 않을 가능성이 높은 고객을 사전에 예측하는 것.

세부 목표는 다음과 같습니다.

```
고객 결제 정보, 멤버십 정보, 음악 청취 데이터를 통합하여 분석
고객 이탈 여부 예측 모델 구축
LightGBM, XGBoost, Decision Tree 모델 성능 비교
Log Loss 기반 최적 모델 선정
SHAP 및 Feature Importance 분석을 통한 주요 이탈 요인 도출
Streamlit 기반 이탈 예측 대시보드 구현
이탈 위험 고객군 식별 및 활용 방안 제시
```

이를 통해 기업은 이탈 위험 고객에게 맞춤형 할인 쿠폰 제공, 요금제 개선, 고객 유지 마케팅 등의 전략을 수립할 수 있습니다.

### 프로젝트 구조
```

```

---

## 🛠 기술 스택

| Category | Stack |
|----------|-------|
| 💻 Language | ![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| 📊 Data Processing | ![Pandas](https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| 🤖 Machine Learning | ![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBOOST-228B22?style=for-the-badge&logo=xgboost&logoColor=white) ![LightGBM](https://img.shields.io/badge/LIGHTGBM-9ACD32?style=for-the-badge) ![RandomForest](https://img.shields.io/badge/RANDOM_FOREST-228B22?style=for-the-badge) ![DecisionTree](https://img.shields.io/badge/DECISION_TREE-8B4513?style=for-the-badge) |
| 📈 Visualization | ![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-11557C?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |

---

## WBS

1단계: 
| 작업명 | 담당자 | 기간 | 상태 |
|--------|--------|------|------|
| 주제 선정 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |
| 데이터 탐색 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |


2단계: 
| 작업명 | 담당자 | 기간 | 상태 |
|--------|--------|------|------|
| 데이터 수집 및 전처리 | 김봉남, 김효민, 김세희 | 2026-06-02 | ✅ |
| feature engineering | 안영선 | 2026-06-02 | ✅ |
| 모델 개발 및 훈련 | 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |
| 최종 모델 선정 및 시각화 | 안영선, 김봉남 | 2026-06-02 | ✅ |


3단계: 
| 작업명 | 담당자 | 기간 | 상태 |
|--------|--------|------|------|
| 최종 모델 선정 및 시각화 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-04 | ✅ |
| 산출물 정리 | 안영선, 유진영, 김효민, 김세희 | 2026-06-04 | ✅ |
| 발표준비 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-04 | ✅ |


---

## 요구사항 정의서

| 항목 | 구분 | 설명 |
| :---: | :---: | :--- |
| |  |  |

---

## 데이터 수집
### 데이터 출처
- WSDM - KKBox's Churn Prediction Challenge
    - https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

- 참고 논문 
    - KKBox Churn Rate Prediction: A Special Focus on Imbalanced Dataset

## 데이터 분석 및 전처리
### 1. 데이터 설명
### 2. 데이터 분석
### 3. 데이터 전처리



## 주요 프로시저
### 1. 주요 함수

[메인_페이지.py]
- 

### 2. 
- 

--- 

## 수행 결과
### 1) 메인 페이지 - 
<img src="images\main_page.png">



## 한 줄 회고
#### 안영선
 -

#### 김봉남
 -

#### 유진영
 -

#### 김효민
 - 

#### 김세희
 - 
 
