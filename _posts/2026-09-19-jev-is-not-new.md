---
title: "Jev 很新？rerank 表示：我一直都是這樣"
date: 2026-09-19
tags: [llm-architecture, rerank, chain-of-thought]
description: "Jev 真正的賣點是選項到 runtime 才給，但 rerank 本來就是這樣。順便問一句：如果不寫思維鏈也一樣準，那思維鏈到底在幹嘛？"
---

TypeSafe AI 在 2026 年 9 月推出了 Jev，並宣布這是一個新的模型類別，叫做「System One」。發表文的形容詞很多：不會幻覺、快 40–200 倍、便宜 40–400 倍、機率經過校準。

看完之後我只有一個感想：這東西我們是不是見過？

## 1. 先把宣傳詞撥掉

**「不會幻覺」**：Jev 的輸出被 schema 限定在事先列好的選項裡，它當然吐不出選項以外的東西。這是介面設計保證的，不是模型的能力。這就好比說選擇題不會寫錯字。它還是可以選錯，而且可以很有自信地選錯。

**「快又便宜」**：這是工程上的成果，值得肯定，但不是觀念上的新東西。

撥掉這些之後，剩下真正的賣點只有一個：

> **問題和選項都是在呼叫的當下才定義的，模型不用重新訓練，就能直接給出各選項的機率分佈。**

傳統的分類模型做不到這件事。BERT 接一個分類頭，標籤在訓練時就固定了，想加一個類別就得重新訓練。Jev 則是今天丟給它「這張票是 bug、feature 還是 question」，明天丟「這筆交易詐騙的風險是低、中還是高」，它都能接。

聽起來很厲害。但問題是……

## 2. rerank 一直都是這樣

rerank 的工作是：給一個 query 和一堆候選文件，替每份候選打分數。重點在於，**候選文件也是 runtime 才給的**。reranker 在訓練時從沒看過你明天要丟給它的那批文件，它照樣能打分數。

把兩邊的介面對齊來看：

| | Jev | rerank |
|---|---|---|
| 輸入 | state + 問題 | query |
| runtime 才給的東西 | 選項 | 候選文件 |
| 輸出 | 每個選項的機率 | 每份候選的分數 / 排序 |

Jev 做的，說穿了就是對一堆很短的候選（標籤）做 rerank。

連它最得意的「一次問 1,500 題，也只要幾百毫秒」，拆開來看也不過是這樣：state 只算一次，所有問題共用這段前綴，再各自對自己的選項做 rerank。批次 rerank 共用 query 前綴，本來就是這樣省時間的。（延遲數據見文末 Appendix。）

這條路也不是 rerank 才有。2019 年就有人把 zero-shot 分類寫成 NLI 問題：對每個標籤問模型「這段文字是否在講 X」，讓它一對一打分數（[Yin et al., 2019](https://arxiv.org/abs/1909.00161v1)）。標籤就是候選，分類就是 rerank，這個觀念至少七年了。

順帶一提，這整件事也可以看成業界繞了一大圈，又回到 discriminative model。LLM 流行之後，大家什麼問題都丟給生成模型，連分類都要它「生成」一個標籤。Jev 等於是走回頭路，然後給這條路取了一個新名字。

## 3. 在 Jev 之前，我們是怎麼做的

沒有 Jev 的時候，要讓 LLM 在一組 runtime 選項裡做決定，標準做法是 **structured output**：給一個 JSON schema，叫模型照格式填。講究一點的，還會在答案前面塞一個 `reasoning` 欄位，讓它先想一想再回答。

成本先不提。這條路本身就有個問題：**格式限制會傷害推理能力**。[Tam et al.（2024）](https://arxiv.org/abs/2408.02442v3)發現，格式要求越嚴格，推理任務的表現掉得越多；而且這不是解析失敗造成的，有些模型的解析錯誤率幾乎是零，表現照樣下滑。也就是說，我們一邊叫模型思考，一邊又用格式把它綁住。

但同一篇論文還有後半句：**在分類任務上，格式限制反而會提高準確率**。推理任務被格式綁住會變差，分類任務被格式綁住卻會變好。而 Jev 賣的正是分類：它把格式限制推到極致，連一個字都不讓模型生成。

這就引出一個有趣的問題。

## 4. 如果 Jev 一樣準，思維鏈還重要嗎？

Jev 不寫思維鏈，只做一次 forward pass 就直接給機率。如果它在這類決策任務上，跟會先寫一大段推理的 LLM 一樣準，甚至只是接近，那我們這幾年花在思維鏈上的 token，到底買到了什麼？

### 很遺憾，官方的比較回答不了這個問題

把 TypeSafe 公開的比較一個一個看過去（[發表文](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[官方文件](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook)）：

- **首頁那個「快 193.6 倍、便宜 444.6 倍」**，來自他們自己設計的 workflow evals。這裡的「參考答案」不是真實標籤，而是 GPT-6 Astra 和 Fable 5.1 兩個模型預測的平均。TypeSafe 自己也承認這會讓結果偏向 OpenAI 和 Anthropic 的模型。
- **跟 GPT-5.6 Terra（預設 reasoning）的並排 demo**：只有一題答案不一樣。但這只是一個例子。
- **有一個 demo 把對手的 reasoning 關掉了**，TypeSafe 自己補了一句：開了 reasoning 的 LLM 會好很多。
- **官方文件裡確實有拿 reasoning 模型比的實驗**（gpt-5.5、claude-opus-4-8），但比的是同一題重跑 15 次答案穩不穩定。文件自己寫得很清楚：這不代表準確率，也不代表誰比較強。

所以整理下來：「不思考的 Jev」對上「會思考的 LLM」、用真實答案評分，這樣的比較**一個都沒有**。能看到的，只有「Jev 跟大模型的答案有多一致」。

如果那兩個參考模型有開 reasoning（發表文沒寫），那「跟它們答案一致」本身就暗示：在這類任務上，思考可能沒改變什麼答案。只是官方數據證明不了這一點，我們得去別的地方找。

### 不過，這個問題的答案其實早就有了

[Sprague et al.（2024）](https://arxiv.org/abs/2409.12183v3)整理了一百多篇用到 CoT 的論文，自己又在 14 個模型、20 個資料集上實測。結論就寫在標題上：**CoT 的好處主要集中在數學和符號推理**，其他任務幾乎沒差。在 MMLU 上，CoT 帶來的進步有高達 95% 來自題目或答案裡含有「=」的題目。

Jev 主打的是什麼任務？分類、路由、風險判斷。這些剛好都是 CoT 本來就幫不上什麼忙的類型。

外部的探測結果也對得上。archerhume 用自己出的數學題測 Jev，發現模冪運算這類題目的平均信心只有 35%，實際卻答對 56%，校準完全跑掉（[archerhume, 2026](https://archerhume.com/posts/jevs-architecture-unmasked/)）。CoT 最有用的領域，剛好就是 Jev 連自己有幾成把握都算不準的地方。

### 就算推理重要，它也可以被壓進權重裡

有人可能會說：Jev 說不定是拿 reasoning 模型的結果訓練出來的，推理只是事先被攤進了權重（amortized reasoning），所以不能說推理不重要。

但這個反駁反而把結論推得更遠：**如果推理在推論時可以拿掉，或者可以直接蒸餾、跳過，那就代表對神經網路來說，寫出推理步驟根本不是必要的。**推理步驟頂多是訓練時的鷹架，房子蓋好就可以拆了。

這不是空想。[Deng et al.（2024）](https://arxiv.org/abs/2405.14838v1)先用顯式 CoT 訓練模型，再逐步移除中間步驟、讓模型把它們內化。結果 GPT-2 Small 不寫任何中間步驟，就能以 99% 的準確率算出 9×9 乘法；同樣的模型用一般方式訓練，連 4×4 都做不到。Mistral 7B 在 GSM8K 上不輸出任何步驟，也能拿到 50% 以上。

至於 CoT 寫出來的那些步驟，本來就未必是模型真正的依據。[Turpin et al.（2023）](https://arxiv.org/abs/2305.04388v2)發現，在 few-shot 範例裡把正確答案一律排在 (A)，模型就會偏向選 (A)，但它寫出來的推理完全不會提到這件事，只會給一段看起來很合理的理由。

### 好，但要講清楚邊界

「推理不重要」有一個理論上的邊界。[Merrill & Sabharwal（2024）](https://arxiv.org/abs/2310.07923v5)證明，中間步驟確實會擴張 transformer 的計算能力：步數越多，能解的問題類別越大；允許多項式步數時，可解的問題剛好就是多項式時間可解的那一類。

直觀地說，一次 forward pass 的計算深度是固定的。問題規模固定的時候，例如 9×9 乘法，推理可以被壓進權重；但如果需要的串行步數會跟著輸入一起變長，固定深度遲早不夠用，這時候中間步驟就是必要的。

有趣的是，TypeSafe 自己的做法剛好示範了這一點。他們建議把複雜的判斷拆成很多個「窄而具體」的小問題，再用程式碼把答案串成一個 workflow；他們也說，讓 LLM 把所有邏輯塞進同一段思維鏈裡，表現明顯比較差。換句話說，串行的步驟並沒有消失，只是從模型的思維鏈**搬到了程式碼的控制流程裡**。每一次模型呼叫都只負責一個淺的判斷，深度由程式碼提供。

所以比較精確的說法是：

> **對大多數「System One」任務來說，寫出推理步驟本來就不重要，這件事 2024 年就知道了。**
>
> **推理真正不可或缺的地方，是需要的串行步數會隨問題規模增長的任務。**

又一個不新鮮的發現。

## 5. 校準：好奇它怎麼做，但大概逃不出老方法

Jev 最讓人好奇的地方，其實是它宣稱的校準：信心越高，正確率越高。rerank 通常做不到這件事，因為 ranking loss 只要求排序正確，不管分數本身有沒有意義。一個排序全對的 reranker，可以把正例都打 0.55、負例都打 0.52，但 0.55 並不代表「55% 會對」。

TypeSafe 說他們的訓練方法叫 RLCD（Reinforcement Learning for Calibrated Decisions）。[官方文件](https://docs.typesafe.ai/introduction/machine-learning-primer)花了不少篇幅解釋「校準」是什麼意思：給 0.2 的事情，大約 20% 會發生；給 0.8 的，大約 80% 會發生。至於怎麼做到的，只有一句「probabilities are optimized against outcomes」。用什麼 loss、用什麼資料、怎麼驗證，都沒有寫。

名字很新，但要讓機率數字本身有意義，工具箱大概還是傳統 ML 那一套：

- 用 **proper scoring rule**（log loss、Brier score）搭配真實結果訓練。這類損失函數只有在回報的機率等於真實機率時才會最小（[Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437)）。
- 訓練完再做 **temperature scaling** 之類的事後修正（[Guo et al., 2017](https://arxiv.org/abs/1706.04599v2)）。
- 用 **reliability diagram**、ECE 來驗證。

archerhume 量到的 ECE 是 0.0313（MMLU 抽樣 1,200 題），看起來很漂亮。但他自己也指出，這 1,200 題裡有 990 題落在 0.9–1.0 這一箱：預測 98.7%，實際 96.3%。八成以上的題目都是模型很有把握的簡單題，ECE 當然好看。換成它不熟的數學題，就是前面那個 35% 對 56%。

所以要不要信 Jev 的機率，老方法最可靠：拿你自己的任務、你自己的標準答案，畫一張 reliability diagram。

## 結論

- Jev 的核心價值是 runtime 才給選項，而 rerank 一直都是這樣。
- Jev 不寫思維鏈也能做決策，這一點早就被預測到了：CoT 本來就只在數學和符號推理上有明顯幫助。
- Jev 的校準如果是真的，那是它最有價值的部分，但方法大概也不會是新的。

Jev 可能是一個很好用的產品。它只是不新。

真正值得記下來的，也許是另一件事：我們花了好幾年，把分類問題硬塞給生成模型，還叫它先寫一篇作文再回答。現在有人把它拆回來，大家反而覺得這是創新。

順著這個問題往下還有兩篇：既然 Jev 是 rerank，那[為什麼不能做成 embedding](/posts/decomposition-and-measure-zero/)；以及把維度一路推到無限之後，[會發生什麼事](/posts/infinite-width-fireworks/)。

---

## Appendix：Jev 的架構推測

TypeSafe 沒有公開 Jev 的架構。以下整理自 [archerhume, *Jev's Architecture Unmasked*（2026）](https://archerhume.com/posts/jevs-architecture-unmasked/)，作者打了一萬多次 API 做黑箱探測。**作者自己也強調，這份重建有相當程度是推測。**

**(a) 問題數增加，延遲幾乎不變**

| 問題數 | 1 | 100 | 1,000 | 1,500 |
|---|---|---|---|---|
| 中位延遲 | 86.5 ms | 82 ms | 453.5 ms | 610 ms |

逐 token 生成不可能這樣。比較合理的解釋是 state 只編碼一次、所有問題共用。

**(b) `output_tokens` 跟延遲對不上**

一個有 255 個選項的回應，帳單記了 2,714 個 output token，延遲卻跟簡單的請求差不多；而且這個數字會隨著模型根本看不到的 request metadata 改變。可見它是事後序列化算出來的計費數字，背後沒有真的在解碼。

**(c) 違反 IIA（independence of irrelevant alternatives）**

在一個四選項的理賠原因分類題裡，加上一個不相干的第五個選項「壞天氣造成的」，原本兩個選項之間的 log-odds 從 +0.49 掉到 +0.08。重做 10 個隨機化區塊，平均變化 −0.28，95% 信賴區間 [−0.36, −0.19]，每一區都同方向。

如果每個選項的分數 $$z_k$$ 是各自獨立算好、最後才一起過 softmax，那

$$
\frac{p_i}{p_j} = \frac{e^{z_i} / \sum_k e^{z_k}}{e^{z_j} / \sum_k e^{z_k}} = e^{z_i - z_j}
$$

分母會約掉，加選項不該改變既有選項之間的比值。實際上卻變了，代表選項在打分時彼此看得到，也就是 listwise 的處理方式。

**作者的結論**

作者的推測是：一個被改造來做決策的 causal transformer（可能是 sparse MoE），state 只編碼一次，各個問題在獨立的分支裡處理、彼此看不到，但同一題的選項會在做選擇之前互相交互，最後由輸出頭直接讀出機率，不生成文字。講到選項交互時，他拿 FIRST（[Reddy et al., 2024](https://arxiv.org/abs/2406.15657v1)）來類比：FIRST 也是從第一個 token 的 logits 直接讀出 listwise 排序，不逐 token 生成。作者也明說，backbone 的部分是整個推測裡最不確定的。

## 參考資料

- TypeSafe AI (2026). [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- TypeSafe AI (2026). [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)、[Self-consistency: choices](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook).
- archerhume (2026). [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/).
- Yin, W., Hay, J., & Roth, D. (2019). [Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach](https://arxiv.org/abs/1909.00161v1).
- Tam, Z. R., et al. (2024). [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/abs/2408.02442v3).
- Sprague, Z., et al. (2024). [To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning](https://arxiv.org/abs/2409.12183v3).
- Deng, Y., Choi, Y., & Shieber, S. (2024). [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838v1).
- Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388v2).
- Merrill, W., & Sabharwal, A. (2024). [The Expressive Power of Transformers with Chain of Thought](https://arxiv.org/abs/2310.07923v5).
- Gneiting, T., & Raftery, A. E. (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437).
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599v2).
- Reddy, R. G., et al. (2024). [FIRST: Faster Improved Listwise Reranking with Single Token Decoding](https://arxiv.org/abs/2406.15657v1).
