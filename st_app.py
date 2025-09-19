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

uploaded_files = st.file_uploader("画像を複数アップロードしてください...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 類似画像を検索する
if uploaded_files:
    st.write("アップロードされた画像:")
    query_images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        query_images.append(image)

    # 複数画像のベクトルを計算し、平均ベクトルを作成
    all_query_features = []
    for image in query_images:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        all_query_features.append(image_features)

    # 平均ベクトルを計算
    if all_query_features:
        query_features_tensor = torch.cat(all_query_features)
        query_features_mean = query_features_tensor.mean(dim=0, keepdim=True)
        
        # コサイン類似度を計算
        similarities = F.cosine_similarity(query_features_mean, feature_tensor)
        best_match_index = torch.argmax(similarities).item()
        
        # 結果を表示
        result_image = all_images[best_match_index]
        st.write("検索が完了しました！")

        st.image(result_image, caption=f"最も似ている画像 (類似度: {similarities.max():.4f})", use_column_width=True)
