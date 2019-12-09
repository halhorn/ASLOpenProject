# YouTube 8M Dataset Projects

YouTube 8M Datasetを使って、新規のデータに対してラベルを提案する

## プロジェクト背景


## データについて

### 使用するデータセット

- [YouTube 8M Dataset](https://research.google.com/youtube8m/)


### データの保存場所

- asl-mixi-project-bucket/data/youtube-8m/train/
- asl-mixi-project-bucket/data/youtube-8m/valid/
- asl-mixi-project-bucket/data/youtube-8m/test/

- asl-mixi-project-bucket/data/youtube-8m-frame/train/
- asl-mixi-project-bucket/data/youtube-8m-frame/valid/
- asl-mixi-project-bucket/data/youtube-8m-frame//test/


## モデルについて

### 入出力

### 性能評価方針

## フロントエンド

必要な機能

- 動画を投稿して表示する
- 投稿された動画から特徴量を取得する
- 特徴量の結果を作成したモデルで推論
- 推論の結果を画面に返す

### 投稿サイト

- python flask + vue がとりあえずの候補

### 特徴量抽出

- mediapipeにシステムがあったのでこれを動かす
  - https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/youtube8m
  - docker imageが公開されているのでgke上に構築する
