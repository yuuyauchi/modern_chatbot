# 🦙🦙🦙ModernChatBot🦙🦙🦙

txtファイル、htmlファイル等の様々な形式のファイルデータを学習データとして使用できるLLMが完成形。

## 概要
langchainやllamaindex等を用いた再学習により、任意の情報に関する受け答えが可能なチャットボットを作成いたしました。

## ディレクトリ構成
```
chatbot_project/
│
├── data/
│   ├── raw/              # url, textを含むあらゆるデータを保存
│   ├── processed/        # テンプレート等の学習データの保存場所
│   └── models/           # オフラインLLMなどの保存場所
│
├── src/
│   ├── preprocessing/    # データの前処理に関するスクリプト
│   ├── models/           # モデルの定義や学習に関するスクリプト
│   ├── utils/            # 便利な関数やユーティリティを格納
│   └── main.py           # 実行スクリプトやAPIのエンドポイント定義
│
├── configs/              # 設定ファイルやハイパーパラメータの定義
│
├── logs/                 # ログファイルの保存先
│
├── tests/                # テストコード
│
├── docs/                 # ドキュメントや使用方法
│
└── README.md
```
## 使用方法

①必要であれば以下のコマンドでLLMの学習を実行し、モデルを保存。
```
python src/train.py
```

②以下のコマンドで学習したモデルを用いて、chatbot用のUIをローカルサーバー上に作成する。
```
streamlit run run.py
```