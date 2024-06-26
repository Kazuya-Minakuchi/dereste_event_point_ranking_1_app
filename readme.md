## 概要

時系列予測の練習用に作った、[アイドルマスターシンデレラガールズスターライトステージのイベント（アタポン形式）](https://imascg-slstage-wiki.gamerch.com/%E3%82%A4%E3%83%99%E3%83%B3%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF#content_2_1)で、1位の人が終了時に何ポイント取るか予測するアプリ

* これまでのイベントデータを保存しておき、次回のイベントの1位を予測する
* 状態空間モデルを用いている

## 使い方

1. src内のcore.pyを実行
2. 「データフレームの表示、編集」「学習・予測」のどちらか選ぶ
   * データフレームの編集
      * データフレーム表示
      * グラフ表示
      * データ追加
      * データ削除
   * 学習・予測
      * 学習&予測
      * 学習結果表示（前回の予測時）
      * 予測結果表示（前回の予測時）

## 注意

* 使用前にPystanをインストールする必要あり
  * pystanインストール：`pip install pystan`
  * `pystan - Unable to find vcvarsall.bat`というエラーが出たら
    * [ココ](http://y-okamoto-psy1949.la.coocan.jp/Python/PyStanWinLinux/)の通り、Visual StadioのC++をインストールする

## ディレクトリ構成

* data：イベントについてのデータ置き場
* models：作成したモデル
* notebooks：検討段階のノートブック
* src：実行ファイル

## できたら改善したいところ（メモ）

* 次回イベントの情報を変えるたびに学習し直しており、時間がかかる
  * 過去イベントの情報が変わらないなら、そこまでのトレンド成分や係数は変わらないはずなので、一回学習したら次回イベントデータ入れたらサクッと予測できるようにしたい
