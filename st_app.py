import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import pickle

# ----- 1. モデルとデータの読み込み -----
# Streamlitのキャッシュ機能を使って、重い処理を初回のみ実行するようにする
@st.cache_resource
def load_model_and_data():
    print("Loading model and data...")
    # デバイスの確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # モデルの読み込み
    model_name = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # 事前計算したベクトルと画像をファイルから読み込み
    feature_tensor = torch.load("features.pt").to(device)
    with open("images.pkl", "rb") as f:
        all_images = pickle.load(f)
        
    print("✅ Load complete!")
    return device, processor, model, feature_tensor, all_images

# 関数を実行して、それぞれの変数を取得
device, processor, model, feature_tensor, all_images = load_model_and_data()


# ----- 2. WebアプリのUI（ユーザーインターフェース）部分 -----
st.title("類似画像検索アプリ")
st.write("画像をアップロードすると、データベースの中から最も似ている画像を探します。")

uploaded_file = st.file_uploader("画像をアップロードしてください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # アップロードされた画像を表示
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="アップロードされた画像", use_column_width=True)
    
    # ----- 3. 検索ロジックの実行 -----
    st.write("検索を実行中...")
    
    # クエリ画像をベクトル化 (Notebookのセル3と同じ処理)
    inputs = processor(images=query_image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_features = model.get_image_features(**inputs)

    # コサイン類似度を計算
    similarities = F.cosine_similarity(query_features, feature_tensor)
    best_match_index = torch.argmax(similarities).item()
    result_image = all_images[best_match_index]
    
    st.write("検索が完了しました！")

    # 結果の画像を表示
    st.image(result_image, caption=f"最も似ている画像 (類似度: {similarities.max():.4f})", use_column_width=True)