# SKN31-2nd-4Team

## 팀 및 팀원 소개

### 팀 명
- <b>영크크Box(kkbox)</b>🎵

### 팀원
| 안영선 | 김봉남 | 유진영 | 김효민 | 김세희 |
| :---: | :---: | :---: | :---: | :---: |
| <a href="https://github.com/dksdudtjs94"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/bongrybong"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/ujneg18-source"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/hyomin0357"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> | <a href="https://github.com/kimsahee0401271111-collab"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> |
|<img src="images\마틴.png" width="150" height="150"> | <img src="images\주훈.png" width="150" height="150"> | <img src="images\제임스.png" width="150" height="150"> | <img src="images\성현.png" width="150" height="150"> | <img src="images\건호.png" width="150" height="150"> |
| <b>PM</b>, 데이터 전처리, readme | <b>부PM</b>, 데이터 전처리, UI/UX | 데이터 전처리, LightGBM 모델링| 데이터 전처리, Decision Tree 모델링 | 데이터 전처리, XGBoost 모델링 |

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


→ 고객 결제 정보, 멤버십 정보, 음악 청취 데이터를 통합하여 분석 <br>
→ 고객 이탈 여부 예측 모델 구축 <br>
→ LightGBM, XGBoost, Decision Tree 모델 성능 비교 <br>
→ Log Loss 기반 최적 모델 선정 <br>
→ SHAP 및 Feature Importance 분석을 통한 주요 이탈 요인 도출 <br>
→ Streamlit 기반 이탈 예측 대시보드 구현 <br>
→ 이탈 위험 고객군 식별 및 활용 방안 제시 <br>

이를 통해 기업은 이탈 위험 고객에게 맞춤형 할인 쿠폰 제공, 요금제 개선, 고객 유지 마케팅 등의 전략을 수립할 수 있습니다.

### 주요기능
본 Streamlit 대시보드는 머신러닝 예측 모델의 결과를 비즈니스 의사결정 및 마케팅 액션과 유기적으로 연결하기 위해 개발되었습니다. 당월 만료 대상자의 규모와 방치 시 증발할 잠재적 매출 리스크를 대만 달러(NT$) 및 한화(KRW) 가치로 실시간 계량화하여 보여주는 산출 기능을 제공합니다. 동시에 LightGBM 모델의 평가지표 매트릭스와 변수 중요도를 확인하여 모델의 설명 가능성(XAI), 즉 고객 이탈의 주원인을 직관적으로 파악할 수 있습니다. <br>
이를 바탕으로 세분화된 위험군별 특성은 물론, 고객의 자동 갱신 여부나 구독 취소 버튼 클릭 행태에 따른 실제 이탈률을 교차 분석함으로써 자동 결제 전환 프로모션과 같은 구체적인 마케팅 처방의 정량적 근거를 제공합니다. 마지막으로 최근 14일간 청취량이 급감한  위험군을 추출하고 , 이들을 그로스 마케팅에 즉시 활용할 수 있도록 맞춤형 전략 및 타겟 오디언스 CSV 명단 다운로드 기능까지 통합적으로 지원합니다.


### 프로젝트 구조
```
SKN31-2nd-4Team
├─ app.py
├─ data
│  ├─ lgbm_feature_importance.csv
│  ├─ test_원본.csv
│  ├─ 성능.csv
│  └─ 최종모델_shap.csv
├─ docs
│  ├─ data_preprocess.md
│  └─ model.md
├─ eda.ipynb
├─ images
│  ├─ screen1.png
│  ├─ screen2.png
│  ├─ screen3.png
│  ├─ screen4.png
│  ├─ screen5.png
│  ├─ 건호.png
│  ├─ 마틴.png
│  ├─ 성현.png
│  ├─ 제임스.png
│  └─ 주훈.png
├─ modeling.ipynb
├─ README.md
├─ requirements.txt
├─ src
│  ├─ data_loader.py
│  ├─ metrics.py
│  ├─ models
│  │  ├─ dt.py
│  │  ├─ lgbm_model.py
│  │  ├─ xgboost_model.py
│  │  └─ __init__.py
│  ├─ preprocessing
│  │  ├─ members 전처리.py
│  │  ├─ transaction 전처리.py
│  │  └─ userlog 전처리.py
│  └─ __init__.py
├─ utils
│  ├─ data_loader.py
│  ├─ headers.py
│  ├─ metric_calculator.py
│  ├─ sidebar.py
│  ├─ style.css
│  ├─ unzip.py
│  └─ 전처리.ipynb
├─ views
│  ├─ tab_eda.py
│  ├─ tab_model.py
│  ├─ tab_solution.py
│  └─ tab_trend.py
└─ 산출물
   ├─ 데이터 전 처리 결과서.md
   └─ 모델 학습 결과서.md

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
|:--------:|:--------:|:------:|:------:|
| 주제 선정 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |
| 데이터 탐색 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |


2단계: 
| 작업명 | 담당자 | 기간 | 상태 |
|:--------:|:--------:|:------:|:------:|
| 데이터 수집 및 전처리 | 김봉남, 김효민, 김세희 | 2026-06-02 | ✅ |
| feature engineering | 안영선, 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |
| 모델 개발 및 훈련 | 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |
| 최종 모델 선정 및 시각화 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-02 | ✅ |


3단계: 
| 작업명 | 담당자 | 기간 | 상태 |
|:--------:|:--------:|:------:|:------:|
| 최종 모델 선정 및 시각화 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-04 | ✅ |
| 산출물 정리 | 안영선, 유진영, 김효민, 김세희 | 2026-06-04 | ✅ |
| 발표준비 | 안영선, 김봉남, 유진영, 김효민, 김세희 | 2026-06-04 | ✅ |


---

## 기능 요구사항

### 데이터 명세 · 분석 · 전처리
- [바로가기](docs/data_preprocess.md)

### 모델링
- [바로가기](docs/model.md)

## 비기능 요구사항
| 항목 | 내용 |
|------|------|
| 재현성 | 모든 모델 `random_state=42` 고정 |
| 모듈화 | 전처리 / 모델별 학습 / 대시보드 파일 분리 |
| 확장성 | 새로운 모델 추가 시 개별 파일만 작성하면 되는 구조 |
| 저장 관리 | 중간 산출물(csv)은 `.gitignore` 처리 |

--- 

## 수행 결과
### 1) 메인 페이지 - 
<img src="images\main_page.png">

---

## 한 줄 회고
#### 안영선
 - 지금까지 배워온 것을 통해 데이터를 다루고 모델을 선정하여 학습과 성능 평가, Streamlit까지 연결하면서 ML 프로젝트의 전체 흐름을 경험해 볼 수 있어서 좋았고 열심히 그리고 잘하는 팀원들을 만나 의미가 있는 프로젝트였습니다.

#### 김봉남
 - 이렇게 대용량의 데이터와 여러 데이터셋을 조인해서 분석해볼 수 있어서 좋았습니다. 본격적인 머신러닝 프로젝트를 어떻게하면 가장 최적으로 진행할 수 있을지 앞으로 더 많이 경험해보고싶습니다. EDA의 중요성에 대해서도 알 수 있었습니다. 앞으로 더 다양한 데이터를 다뤄보고 연습하고자 합니다.

#### 유진영
 - 데이터 전처리부터 모델 평가까지의 전 과정을 직접 겪으며 데이터의 중요성을 뼈저리게 느꼈던 의미 있는 시간이었습니다. 많은 걸 배웠고 이제 컴퓨터 좀 쉬게 해주고 싶습니다 ㅎㅎ

#### 김효민
 - 팀원들의 코드를 참고하면서 실전 데이터 파이프라인의 흐름을 배울 수 있었습니다. 작고 큰 에러들로 조금 더디게 진행되었지만 팀원과 원인을 함께 분석하는 과정이 좋은 경험이 되었습니다. 앞으로 모델의 작동 원리와 한계점을 더 자세히 공부해야겠다는 생각이 들었습니다.

#### 김세희
 - 이번 프로젝트를 통해 데이터 전처리부터 머신러닝 모델 학습, 성능 평가까지의 과정을 직접 경험하며 많은 것을 배울 수 있었습니다. 특히 모델 결과가 예상과 다르게 나왔을 때 팀원들과 함께 원인을 찾고 해결하는 과정에서 데이터의 중요성과 협업의 가치를 느낄 수 있었던 의미 있는 프로젝트였습니다.

---