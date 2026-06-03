# 1. 라이브러리
import os
import pandas as pd
import numpy as np
import shap # SHAP 라이브러리 추가 (!pip install shap 필요)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, log_loss
)

# ==========================================
# 폴더 설정 부분
# ==========================================
RES_DIR = './results'
os.makedirs(RES_DIR, exist_ok=True)
print(f"📁 저장 경로 설정 완료: {RES_DIR}")

# 2. 데이터 불러오기
try:
    train = pd.read_csv(r"./data/train.csv")     
    test = pd.read_csv(r"./data/test.csv")      
except FileNotFoundError:
    print("❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit() # 데이터가 없으면 즉시 종료

# 3. 사용하지 않을 컬럼 제거
drop_cols = [
    "msno", "is_churn", "registration_init_time",
    "tx_transaction_date", "tx_membership_expire_date", "last_listen_date"
]

train_drop_cols = [c for c in drop_cols if c in train.columns]
test_drop_cols = [c for c in drop_cols if c in test.columns]

X_train = train.drop(columns=train_drop_cols)
y_train = train["is_churn"]

X_test = test.drop(columns=test_drop_cols)
y_test = test["is_churn"]

# 4. bool 컬럼을 0/1로 변환
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# 5. Decision Tree 모델 생성 및 학습
print("⏳ 모델 학습 중...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

# 6. 예측 및 확률 추출
y_pred = dt_model.predict(X_test)
y_proba = dt_model.predict_proba(X_test)[:, 1] 

# ==========================================
# [파일 1] 모델 성능 (model_performance.csv)
# ==========================================
perf_df = pd.DataFrame([{
    'Model'   : 'Decision Tree',
    'AUC'     : round(roc_auc_score(y_test, y_proba), 6),
    'F1'      : round(f1_score(y_test, y_pred, zero_division=0), 6),
    'Log Loss': round(log_loss(y_test, y_proba), 6),
}])
perf_df.to_csv(f'{RES_DIR}/dt 성능.csv', index=False, encoding='utf-8-sig')
print("✅ [1/4] 'dt 성능.csv' 저장 완료")

# ==========================================
# [파일 2] Feature Importance (importance_decision_tree.csv)
# ==========================================
imp_df = pd.DataFrame({
    'Feature'   : X_train.columns,
    'Importance': dt_model.feature_importances_, 
}).sort_values('Importance', ascending=False).reset_index(drop=True)

imp_df.to_csv(f'{RES_DIR}/dt_feature_importance.csv', index=False, encoding='utf-8-sig') 
print("✅ [2/4] 'dt_feature_importance.csv' 저장 완료")

# ==========================================
# [파일 3] SHAP (shap_summary.csv)
# ==========================================
print("⏳ SHAP Value 계산 중...")
SEED = 42
N_SHAP = min(5000, len(X_test))
X_shap = X_test.sample(N_SHAP, random_state=SEED).reset_index(drop=True)

explainer   = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X_shap)

if isinstance(shap_values, list):
    shap_values_target = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_target = shap_values[:, :, 1]
else:
    shap_values_target = shap_values

shap_df = pd.DataFrame({
    'Feature'      : X_train.columns,
    'mean_abs_shap': np.abs(shap_values_target).mean(axis=0),
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_df.to_csv(f'{RES_DIR}/dt_shap.csv', index=False, encoding='utf-8-sig')
print("✅ [3/4] 'dt_shap.csv' 저장 완료")

# ==========================================
# [파일 4] 이탈률 예측 결과 (churn_prediction_test.csv)
# ==========================================
result = pd.DataFrame({
    'msno'             : test['msno'].values,
    'actual_churn'     : y_test.values,
    'predicted_churn'  : y_pred,
    'churn_probability': y_proba,
})

# ── 10분위 위험 등급 
result['risk_decile'] = pd.qcut(
    result['churn_probability'].rank(method='first', ascending=False),
    q=10,
    labels=list(range(1, 11))
).astype(int)

# ── 3단계 위험군 
result['risk_group'] = result['risk_decile'].map(
    lambda d: 'High Risk' if d <= 3 else ('Mid Risk' if d <= 7 else 'Low Risk')
)

# 이탈 확률 높은 순으로 정렬
result = result.sort_values('churn_probability', ascending=False).reset_index(drop=True)

result.to_csv(f'{RES_DIR}/dt_churn_prediction.csv', index=False, encoding='utf-8-sig')
print("✅ [4/4] 'dt_churn_prediction.csv' 저장 완료")
print("🎉 모든 작업이 성공적으로 끝났습니다!")