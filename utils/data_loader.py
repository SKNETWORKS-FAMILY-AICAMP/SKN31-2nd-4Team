# utils/data_loader.py
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_DIR / "test_원본.csv")
    
    # 3. 모델링 산출물 서브 데이터셋 로드
    try:
        df_shap = pd.read_csv(DATA_DIR / "최종모델_shap.csv")
    except:
        df_shap = pd.DataFrame()
    try:
        df_feat = pd.read_csv(DATA_DIR / "lgbm_feature_importance.csv")
    except:
        df_feat = pd.DataFrame()

    try:
        df_perf = pd.read_csv(DATA_DIR / "성능.csv")
    except:
        df_perf = pd.DataFrame()

    return {
        "user": df,
        "shap": df_shap,
        "feat": df_feat,
        "perf": df_perf
    }