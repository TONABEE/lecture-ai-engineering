import streamlit as st
import pandas as pd
import numpy as np
import time
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# ============================================
# ページ設定
# ============================================
st.set_page_config(
    page_title="ディベート デモ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# ============================================
# タイトルと説明
# ============================================
st.title("ディベートメンターBOT デモ")
st.markdown("これを使ってディベート強くなろう")


# ============================================
# サイドバー 
# ============================================
# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "ホーム" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["練習試合", "履歴閲覧", "サンプルデータ管理","エビデンス検査","ホーム"],
    key="page_selector",
    index=["練習試合", "履歴閲覧", "サンプルデータ管理","エビデンス検査","ホーム"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "ホーム":
    ui.display_home_page
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()
elif st.session_state.page == "エビデンス検査":
    ui.display_evidence_page()
elif st.session_state.page == "練習試合":
    if pipe:
        ui.display_pd_page
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [YONABEE]")
