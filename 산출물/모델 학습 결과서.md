# 모델 학습 결과서

## 1. 목적

고객 정보를 기반으로 서비스 이탈 여부(`is_churn`)를 예측한다.

## 2. 사용 모델

Decision Tree, XGBoost, LightGBM 3개 모델을 학습·비교하여 AUC 기준으로 최종 모델을 선정하였다.

- Decision Tree <br>

    Decision Tree는 데이터를 가장 직관적인 스무고개 형태로 분류하여, 예측 결과에 대한 높은 설명력과 시각적 명확성을 제공하는 머신러닝 모델입니다. 특히 어떤 기준으로 유저가 이탈하는지 그 원인을 추적하고 설명할 수 있기 때문에, 이번 프로젝트의 고객 이탈 예측 및 원인 분석 모델로 선정하였습니다.

- XGBoost <br>

    XGBoost는 하나의 트리로 예측하는 것이 아니라 여러 개의 결정트리를 차례대로 학습시키면서 이전 트리가 틀린 부분을 계속 보완해 나가는 모델입니다. 고객 이탈은 단순히 하나의 변수로 결정되는 것이 아니라 이용 기간, 결제 정보, 가입 경로 등 여러 요인이 복합적으로 영향을 주기 때문에 변수들 간의 복잡한 관계를 잘 학습할 수 있는 XGBoost를 선정했습니다. 또한 정형 데이터에서 높은 예측 성능을 보이는 것으로 알려져 있어 고객 이탈 예측에 적합하다고 판단했습니다.

- LightGBM <br>

    LightGBM은 수많은 트리를 결합해 예측의 정확도를 극대화한 앙상블 모델입니다. 기존 단일 트리 모델의 한계였던 동점자 발생 문제를 극복하고, 고객별 이탈 확률을 소수점 단위까지 정교하게 산출할 수 있습니다. 학습 속도가 매우 빠르면서도 미세한 위험도 차이까지 분류해낼 수 있어, 이탈 예측 및 정밀 타겟팅 모델로 도입하였습니다


## 3. 최종 선정 모델

**최종 선정 모델 : LightGBM**

선택 이유

- 3개 모델 중 f1 score, log loss 성능이 괜찮음
- Feature Importance / SHAP 기반 변수 중요도 확인 가능
- 리프 기반 분기 + Early Stopping으로 효율적 학습

## 4. 학습 데이터

| 항목 | 내용 |
|---|---|
| 학습 데이터 | `train_split.pkl` (516,580 행) |
| 검증 데이터 | `val_split.pkl` (172,193 행) |
| 평가 데이터 | `test_split.pkl` (172,194 행) |
| 타겟 변수 | `is_churn` (0=유지, 1=이탈) |
| 이탈율 | 약 9.46% (클래스 불균형) |
| 사용 피처 수 | 32개 |
| 불균형 대응 | `scale_pos_weight = 9.57` / `class_weight='balanced'` |

## 5. 모델 하이퍼파라미터

튜닝 방법 : GridSearchCV (5-fold, AUC 기준)

**Decision Tree**
- max_depth = 15
- min_samples_leaf = 100
- min_samples_split = 10
- class_weight = balanced

**XGBoost**
- max_depth = 6
- learning_rate = 0.1
- subsample = 0.8
- colsample_bytree = 0.8
- scale_pos_weight = 9.57
- n_estimators (best) = 999

**LightGBM (최종)**
- num_leaves = 127
- learning_rate = 0.1
- min_child_samples = 20
- subsample = 0.7
- scale_pos_weight = 9.57
- early_stopping = 50 → best_iteration = 1164

## 6. 성능 평가 결과

| Model | AUC | F1 Score | Log Loss |
|---|---|---|---|
| **LightGBM** | **0.9927** | **0.8752** | **0.0733** |
| XGBoost | 0.9918 | 0.8568 | 0.0915 |
| Decision Tree | 0.9691 | 0.6859 | 0.2154 |

## 7. Feature Importance 분석

모델별 Feature Importance는 `results/`에 저장 (`lgbm_feature_importance.png`, `xgb_feature_importance.png`, `dt_feature_importance.png`).

최종 모델(LightGBM)의 SHAP 기반 상위 중요 변수 :

| 순위 | 변수 | mean(\|SHAP\|) |
|---|---|---|
| 1 | expire_day | 0.8109 |
| 2 | is_auto_renew | 0.7511 |
| 3 | expire_weekday | 0.6662 |
| 4 | trans_day | 0.6378 |
| 5 | is_cancel | 0.3960 |
| 6 | reg_time_num | 0.3747 |
| 7 | payment_method_id | 0.3275 |
| 8 | trans_count | 0.3038 |
| 9 | trans_weekday | 0.2834 |
| 10 | city | 0.2112 |

상위 변수가 대부분 **구독 만료일(expire_day) · 자동갱신 여부(is_auto_renew) · 취소 여부(is_cancel)** 등 결제·구독 관련 변수로, 이탈 예측에 결정적 영향을 미친다.

## 8. SHAP 분석

SHAP 분석을 통해 모델 예측에 가장 큰 영향을 주는 변수를 확인하였다. (샘플 5,000건, TreeExplainer)

산출물
- `shap_summary.png` — 변수별 평균 SHAP 영향도 (Bar)
- `shap_beeswarm.png` — 변수값에 따른 SHAP 분포 (Beeswarm)
- `shap_summary.csv` — 변수별 mean(|SHAP|) 수치

`is_auto_renew=0`(자동갱신 미사용), `is_cancel=1`(취소)일수록 이탈 확률을 높이는 방향으로 작용하였다.

## 9. 결론

3개 모델 비교 결과 **LightGBM**을 최종 모델로 선정하였다 (Test AUC 0.9927, F1 0.8752, Log Loss 0.0733). Feature Importance 및 SHAP 분석을 통해 구독 만료·자동갱신·취소 관련 변수가 주요 이탈 요인임을 확인하였다.

추가로 Risk Decile 분석을 통해 예측 확률 기준 위험군을 분류하였으며, 고위험군(High Risk)의 실제 이탈율은 31.5%로 전체 이탈율(9.46%) 대비 약 3.3배 높아 모델의 변별력을 확인하였다.

| 위험군 | 회원수 | 실제 이탈율 |
|---|---|---|
| High Risk | 51,658 | 31.47% |
| Mid Risk | 68,878 | 0.05% |
| Low Risk | 51,658 | 0.00% |
