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

query_images = []
weights = []

# 類似画像を検索する
if uploaded_files:
    st.write("アップロードされた画像:")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        
        # 複数画像を横並びに表示するためのカラムを作成
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(image)
        with col2:
            # スライダーを追加して、重みをユーザーに設定させる
            weight = st.slider(
                "この画像の重要度", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05, 
                key=f"slider_{uploaded_file.name}" # 一意なキーを設定
            )
        
        query_images.append(image)
        weights.append(weight)


# 重み付きの平均（重心ベクトル）を計算
if query_images:
    # まず、重みリストをテンソルに変換
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # 重みの合計が0の場合の処理
    if weights_tensor.sum() == 0:
        st.warning("重みがすべて0です。少なくとも1つの画像の重みを1以上に設定してください。")
        st.stop()  # ここでStreamlitアプリの実行を停止する
    
    all_query_features = []
    for image in query_images:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        all_query_features.append(image_features)

    weighted_features = [feat * weight for feat, weight in zip(all_query_features, weights_tensor)]
    weighted_sum = torch.sum(torch.stack(weighted_features), dim=0)
    query_features_centroid = weighted_sum / weights_tensor.sum()
    
    similarities = F.cosine_similarity(query_features_centroid, feature_tensor)
    best_match_index = torch.argmax(similarities).item()
    
    result_image = all_images[best_match_index]
    st.write("検索が完了しました！")
    st.image(result_image, caption=f"最も似ている画像 (類似度: {similarities.max():.4f})", use_column_width=True)