# イオン北海道公式X SNS分析 — 先行研究レビューと研究設計

Status: **Future research / Backlog**  
Tracking issue: https://github.com/hiromu2001/Mr.-Variable-Picker/issues/1

## 1. 目的

イオン北海道株式会社の公式Xアカウントを対象に、**どのような投稿内容・表現・媒体・投稿条件が、いいね／リポスト／返信などのエンゲージメントと関連するか**を、統計分析・自然言語処理（NLP）・機械学習で検証する。

単なる「伸びた投稿ランキング」ではなく、先行研究に基づく変数設計、カウントデータ回帰、探索的NLP、予測モデル、時系列検証を組み合わせる。

実装開始時には専用リポジトリへ切り出し、データ取得・分析・レポートを分離することを推奨する。

---

## 2. 先行研究から得られる根拠

### 2.1 ブランド投稿のエンゲージメント要因

企業SNS研究では、投稿の以下の特徴とエンゲージメントの関係が繰り返し検討されている。

- vividness：画像・動画等
- interactivity：質問、CTA、リンク等
- informative / entertaining / promotional content
- 投稿曜日・時間
- 文章長
- 言語スタイル

主な先行研究：

1. de Vries, Gensler & Leeflang (2012), *Journal of Interactive Marketing*  
   355件のブランド投稿を対象に、vividness / interactivity / content と likes/comments の関係を分析。  
   DOI: https://doi.org/10.1016/j.intmar.2012.01.003

2. Menon et al. (2019), *Journal of Air Transport Management*  
   FacebookとTwitterの企業投稿を比較し、Twitterでも娯楽性・情報性・販促性・画像等が反応と関連。  
   DOI: https://doi.org/10.1016/j.jairtraman.2019.05.002

3. Deng et al. (2021), *Electronic Commerce Research and Applications*  
   15,396件・104ブランドの投稿について、emotionality / complexity / informality と likes / shares / comments の関係を分析。  
   DOI: https://doi.org/10.1016/j.elerap.2021.101068

4. Message content features and social media engagement (2020), *Journal of Product & Brand Management*  
   CTAやmedia richnessを負の二項回帰で分析。  
   DOI: https://doi.org/10.1108/JPBM-09-2018-2014

### 2.2 日本の企業Xに最も近い先行研究

Tanaka & Huang (2024) は、日本のSHARP公式Twitterについて**500投稿**を対象に、以下をコード化して分析している。

- 画像 / 動画
- リンク
- CTA
- 質問
- brand account personality
- informative / entertaining / promotional
- 曜日
- 文字数
- フォロワー数

目的変数は likes / retweets / replies で、Poisson回帰を使用。

Tanaka, Y. & Huang, L. (2024), “Enhancing social media engagement in Japan: An empirical study of design and content factors on brand account”, *International Journal of Marketing & Distribution*, 27(1-2), 53-72.  
DOI: https://doi.org/10.5844/jsmd.27.1-2_53

今回のイオン北海道分析では、この研究を**最も近いベースライン**とする。

設計上のポイント：

- engagementを一つに合算せず、**likes / reposts / replies を別々に分析**する。
- 画像・動画（vividness）、CTA・質問・リンク（interactivity）、情報・娯楽・販促の内容カテゴリを基本変数にする。
- 人手分類をする場合は、二名コーダー＋Cohen’s kappaで再現性を確認する。

### 2.3 小売・食品領域

小売・食品では、画像・動画・文章長・投稿内容などがエンゲージメントと関連することが報告されている一方、業界・ブランドによって効果は一定ではない。

- Retail brand pages, 2,627投稿：画像/動画や文章長とengagement  
  DOI: https://doi.org/10.1108/IJRDM-09-2018-0195
- Fashion retail：readability / text length / hashtag数  
  DOI: https://doi.org/10.1016/j.jjimei.2022.100067
- Food marketing：computer visionによる画像特徴とengagement  
  DOI: https://doi.org/10.1016/j.jbusres.2022.05.078

したがって、他業界の係数をそのまま期待せず、イオン北海道に固有の変数を追加する。

---

## 3. Research Questions

### RQ1: 投稿内容
どのコンテンツタイプが likes / reposts / replies と関連するか。

候補：
- 商品情報
- 価格・セール・特売
- キャンペーン / プレゼント
- 季節・催事
- 店舗・地域イベント
- 北海道・道産・地域性
- 企業活動 / CSR
- 娯楽・雑談的投稿

### RQ2: 表現方法
文章長、質問、CTA、絵文字、ハッシュタグ、URL、価格表記、感情表現、くだけた表現等は反応とどう関連するか。

### RQ3: メディア
テキストのみ / 画像 / 動画 / GIF で反応は異なるか。画像内テキストや商品写真の特徴まで分析できるか。

### RQ4: 地域性
「北海道」「道産」「札幌」「旭川」「函館」「十勝」「オホーツク」などの地域性を含む投稿は反応と関連するか。

### RQ5: 時系列
曜日、時間帯、季節、年末年始、節分、バレンタイン、新生活、GW、母の日、夏休み等で投稿テーマと反応の関係は変化するか。

### RQ6: 予測
投稿前に利用可能な特徴だけで、将来のエンゲージメントをどこまで予測できるか。

---

## 4. データ設計

### 4.1 取得規模

まず **500投稿のパイロット**。その後、1,000〜3,200投稿を目標。

X User Posts timelineは直近最大3,200投稿を取得可能。より長期の履歴が必要ならFull-Archive Searchを利用する。

公式Docs：
- Timelines: https://docs.x.com/x-api/posts/timelines/introduction
- Full archive search: https://docs.x.com/x-api/posts/search/quickstart/full-archive-search
- Usage/Billing: https://docs.x.com/x-api/fundamentals/post-cap

### 4.2 取得候補フィールド

- post_id
- created_at
- text
- like_count
- retweet_count / repost_count
- reply_count
- quote_count
- media type
- URL / hashtag / mention
- referenced_tweets（reply / quote / repost判定）

### 4.3 Primary analysis対象

原則として**公式アカウントが作成したオリジナル投稿**を対象にし、reply / repostは除外または別分析とする。

キャンペーン・懸賞投稿は反応を極端に押し上げる可能性があるため、以下の感度分析を行う。

1. 全投稿
2. キャンペーン・懸賞除外

---

## 5. 特徴量設計

### A. 機械的に生成できる特徴

- 文字数
- hashtag数
- URL有無
- mention数
- emoji数
- 質問符
- 価格表現（円、¥、%OFF等）
- media type
- 曜日 / 時刻 / 月 / 季節

### B. 先行研究ベース

- informative
- entertaining
- promotional
- remunerative（懸賞・クーポン等）
- CTA
- question
- vividness
- interactivity
- brand account personality / humanized tone

### C. イオン北海道固有

- local_score（北海道・道産・地域名）
- store_specific（店舗名・地域イベント）
- price_promotion
- product_category
- seasonal_event
- corporate_CSR

### D. NLP

- sentence embedding
- sentiment / emotion
- linguistic style（emotionality / informality / certainty）
- BERTopic等による探索的topic

NLPによる自動分類だけで結論を出さず、主要カテゴリは人手コーディングのサンプルで精度確認する。

参考：
- TweetEval: https://doi.org/10.18653/v1/2020.findings-emnlp.148
- BERTopic: https://arxiv.org/abs/2203.05794
- Macanovic (2022), computational text analysis review: https://doi.org/10.1016/j.ssresearch.2022.102784
- Mihalcea et al. (2024), NLP and human behaviour review: https://doi.org/10.1038/s41562-024-01938-0

---

## 6. 統計分析

### 6.1 EDA

- 投稿数の時系列
- likes / reposts / replies / quotes の分布
- 平均・中央値・分散・外れ値
- カテゴリ別箱ひげ / ECDF
- media / 曜日 / 時間帯別比較
- topicの時間推移

### 6.2 カウント回帰

engagementは非負整数で右裾が長くなりやすいため、OLSを主分析にはしない。

- Poisson regression：平均≒分散の場合
- **Negative Binomial regression：過分散がある場合の第一候補**
- zero-inflated / hurdle：ゼロが過剰な場合のみ検討

目的変数は原則として別々にモデル化する。

```text
log(E[likes_i])
  = β0
  + β1 image
  + β2 video
  + β3 CTA
  + β4 entertaining
  + β5 promotion
  + β6 local_score
  + controls
```

Control候補：
- 曜日・時刻
- 月 / 季節
- 文字数
- 長期トレンド
- フォロワー数（投稿時点の履歴が得られる場合）

公開APIだけでは投稿時点のhistorical follower countやimpressionsが得られない可能性がある。社内Analyticsが使える場合は、impressionsをoffsetまたはengagement rateの分母として検討する。

係数は `IRR = exp(β)` も提示して解釈する。

---

## 7. 機械学習

目的は、統計的説明とは別に**投稿前にエンゲージメントを予測できるか**を検証すること。

候補：
- baseline GLM / Elastic Net
- LightGBM / CatBoost
- text embedding + gradient boosting
- 画像特徴を加えたmultimodal model

### 検証

ランダムsplitではなく、**時間順split**を基本とする。

例：
- train: 古い70%
- validation: 次の15%
- test: 最新15%

これにより未来情報の漏洩を減らす。

評価候補：
- MAE
- RMSE（必要に応じてlog1p target）
- Poisson deviance
- Spearman rank correlation

SHAPは予測モデルの説明に使用するが、**因果効果として解釈しない**。

---

## 8. 因果推論上の注意

観察データなので、たとえば

`画像付き投稿 → engagementが増えた`

という因果はそのままでは言えない。投稿テーマ、キャンペーン規模、季節、担当者判断などの交絡が存在する。

Shalizi & Thomas (2011) は観察ネットワークデータでhomophilyとsocial influenceを識別する難しさを示している。  
DOI: https://doi.org/10.1177/0049124111404820

したがって本研究の基本結論は、

- 「〜と関連していた」
- 「〜が予測に寄与した」

までとする。

将来、投稿形式のA/Bテストや自然実験が可能なら因果推論へ拡張する。

---

## 9. 研究倫理・データ管理

- 主対象は企業公式アカウントの公開投稿。
- 一般ユーザーのreply本文はPrimary analysisでは保存しない。
- raw APIデータはGitHubへ直接commitしない。
- X Developer Agreement / Developer Policy / display requirementsを確認する。
- API token、社内Analytics、impressions、clicks等の非公開データはpublic repositoryへ入れない。
- 公開用成果物はpost ID・集約値・派生特徴を中心にし、Xコンテンツの再配布条件を実装前に再確認する。

参考：
- Fiesler & Proferes (2018), Twitter research ethics: https://doi.org/10.1177/2056305118763366
- X policies: https://docs.x.com/developer-terms

---

## 10. ロードマップ

- [ ] Phase 0: X APIアカウント・予算・規約確認
- [ ] Phase 1: 500投稿を取得してraw JSON/CSVをローカル保存
- [ ] Phase 2: EDAとengagement分布確認
- [ ] Phase 3: coding scheme作成（先行研究＋イオン北海道固有項目）
- [ ] Phase 4: 50〜100件を二重コーディングしてCohen’s kappa確認
- [ ] Phase 5: 過分散診断後にPoisson vs Negative Binomialを選択
- [ ] Phase 6: likes / reposts / replies別に推定
- [ ] Phase 7: Embedding / BERTopic / sentimentを探索的追加
- [ ] Phase 8: LightGBM / CatBoost + 時系列holdout
- [ ] Phase 9: SHAP・キャンペーン除外などの感度分析
- [ ] Phase 10: 社内Analyticsが使えるならimpressions / clicks / CTRへ拡張
- [ ] Phase 11: レポート・ダッシュボード化

---

## 11. 本研究の差別化

1. **日本の企業X研究（SHARP 500投稿）の設計をベースラインとして再現**
2. **北海道・地域性・小売・季節販促というイオン北海道固有の特徴を追加**
3. **古典的内容分析＋カウント回帰に、Embedding / topic modeling / gradient boosting / SHAPを接続**

学術的な根拠と、実務で使えるSNS改善示唆の両方を狙う。