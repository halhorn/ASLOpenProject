# Recommendation Project

MovieLensを使ったレコメンデーションプロジェクト


## 使用するデータセット

- MovieLens 20M Dataset
https://grouplens.org/datasets/movielens/20m/


## 開発環境の構築

```bash
# 必要なライブラリーのインストール
pipenv install
```

## airflowの環境構築
以下のようなパイプラインを構築するためairflowを使って該当サービスをオーケストレーションします。

1. GCS or BigQueryに保存してあるデータを抽出しDataFlowで前処理
2. ML Engineにジョブを投げてモデルを学習
3. モデルのデプロイ

初回のみ
```bash
./airflow/script/create_environment.sh
```

dagファイルのデプロイ
```bash
./airflow/script/deploy.sh
```

## デバッグ
airflowで定義した一連のタスクを実行する
```bash
ENVIRONMENT_NAME=YOUR_ENVIRONMENT_NAME
LOCATION=us-central1

gcloud composer environments run ${ENVIRONMENT_NAME} --location ${LOCATION} \
    trigger_dag -- your_dag_name
```
