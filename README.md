# 類似画像検索アプリ

このプロジェクトは、[Streamlit](https://streamlit.io/)を用いた **類似画像検索アプリ** です。
ユーザーが画像をアップロードすると、CLIPモデル (`openai/clip-vit-base-patch32`) を用いて、
事前にベクトル化された画像データベースから最も類似した画像を検索して表示します。

---

## 🔧 セットアップ

1. リポジトリをクローンする
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

2. 必要なライブラリをインストールする
```bash
pip install -r requirements.txt
```

3. 必要なファイルを配置する  
   - `features.pt` … データベース画像の特徴量テンソル
   - `images.pkl` … データベース画像のPIL Imageリスト

4. アプリを実行する
```bash
streamlit run st_app.py
```

---

## 📁 ファイル構成

```
.
├── app.ipynb         # Notebook版。特徴量抽出などの前処理に使用
├── st_app.py          # Streamlitアプリ本体
├── features.pt        # 画像特徴量（ユーザが用意）
├── images.pkl         # 画像リスト（ユーザが用意）
└── requirements.txt   # 依存ライブラリ一覧
```

---

## 📜 ライセンス

このプロジェクトは [MIT License](./LICENSE) のもとで公開されています。
