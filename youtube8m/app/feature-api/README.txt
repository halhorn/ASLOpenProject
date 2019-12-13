## 構成

mediapipeで提供されているyoutube8m用の構成をリモートから叩けるようにapp serverでラップしただけのもの

ただし、mediapipe側でpbファイルの位置等が決め打ちうちになっているので少し差分がある

app server側はsubprocessを使ったかなり力技な仕組み...

## セットアップ

とりあえずmediapiupeをインストールする
https://github.com/google/mediapipe/blob/master/mediapipe/docs/install.md

差分を適応して、appを起動するのみ
