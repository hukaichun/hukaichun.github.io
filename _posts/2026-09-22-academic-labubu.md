---
title: "學術界的 Labubu"
subtitle: "工業級證明產出與知識界的環保意識"
date: 2026-09-22
tags: [commentary, academia, ai, research-practice]
description: "26 位 Fields 獎得主擔心 AI 會切斷數學的傳遞鏈。這個擔憂不是空穴來風：在當代 AI 出現之前，機器學習的 NTK 那條線就沒有被消化過。產出便宜了，垃圾就會堆積；該付帳的不是效率，是用指標的人。"
---

## 1. 數學界的擔憂

9 月 11 日，26 位 Fields 獎得主聯名發表〈[A Severe Misalignment of AI in Mathematics](https://mathandai.org)〉。核心是一句：

> But solving problems is only a tool and proxy for achieving the primary goal of conceptual understanding and insight.

聲明描述了解題之後本來會發生的事：新的方法「would then be studied by a community of mathematicians, through a long and arduous process of talks, discussions, simplifications」，理想的終點是一份研究生、甚至大學生都讀得懂的教科書式陳述。他們擔心的不是 AI 會證錯，而是解答被急著公布，「leaving no time for a proper writeup, the isolation of new methods and ideas, and citing relevant previous work of others」。沒有數學家願意接手，把這些想法「integration into the mathematical canon」，它們就「would never become fully alive」，而「the crucial human transmission chain between mathematicians would be lost」。

Tao 在自己網站上整理的觀點裡，把這件事拆得更細。解一個問題有三個環節：generation、verification、digestion；最後一個是理解它、把它放進脈絡、把它講清楚。過去三者都難，消化是順帶發生的，所以社群只獎勵前兩個。現在 AI 把生成和驗證加速得遠遠超過消化。驗證也救不了這一段：形式驗證保證的是形式命題，「not that it matches intent」。

這個擔憂不是空穴來風。它已經在機器學習這個領域發生過至少一次，而且發生在當代 AI 出現之前。我手邊就有個例子。

[〈兩個 δ〉](/posts/ntk-variational/)重走了一遍 NTK。我不懷疑 Jacot 等人 2018 年的定理丟進 Lean 驗得過，但「權重不動」是那組設定把配比混進了位移的結果，不是網路的性質，他們自己的 kernel 裡，feature 那一項一直都在。之後七年的後續研究，定理大概也都是對的。每一篇指認一個旋鈕：寬度、輸出的縮放、參數化、初始化的大小、層間比例、學習率、優化器；每一個旋鈕都長出一個新名字：lazy、rich、kernel regime、feature learning regime。到了 2021 年，其中最正式的一篇在正文裡寫下：「It may seem somewhat puzzling how the NTK limit induces change in f without feature or feature kernel evolution」。到了 2025 年，二分還在，只是被說成太簡化。沒有被消化的代價，就是七年後大家還在摸同一頭大象，各說各話。

而同一個 $$1/\sqrt n$$ 讓隱藏單元不構成特徵這件事，Neal 1995 年的博士論文就寫了：「with Gaussian priors the contributions of individual hidden units are all negligible, and consequently, these units do not represent "hidden features"」。他寫完就換掉了設定。Jacot 等人為 GP 那一步引了 Neal，那段警告沒有跟著過來。

這段歷史裡沒有人證錯，也沒有人藏東西。每一個結果都生成了、驗證了、發表了；缺的正是聲明點名的那一段：沒有人把它們接到前人後面，收進同一套理論。它們像拆開看一眼就擱在一旁的盲盒，一個接一個堆起來。

## 2. 評分機制的帳

為什麼會這樣？LMU 的博士後 M. Levent Doğan 在回應那份聲明時說得最準（[Proofs and Prompts](https://proofsandprompts.com/2026/09/18/two-responses-to-a-severe-misalignment-of-ai-in-mathematics/)）：

> It would be strange to spend decades constructing institutions that reward output and then blame machines for becoming extraordinarily efficient at producing it.

這筆帳是評分機制的帳。學術界用發表量評價一個人：一篇論文要有新貢獻，最容易拿出來的新貢獻，就是指認一個新的旋鈕、給它一個新的名字。把七年的結果放回同一條式子，不產生新命題，在這套機制裡分數接近零。機器學習那七年，就是這套機制在沒有 AI 的情況下自己跑出來的結果。

Tao 對這個機制的描述更直接：每個人的產出，品質和數量都提高了，整體的訊噪比卻可能下降，他稱之為 Simpson's paradox 的一種。他舉的玩具模型是：生成從六個月縮到一天，仔細的寫作卻只從一個月縮到三週，於是「90% of the literature will now be poorly written」，傳統的審稿會被淹沒。軟體工程已經量到了同一件事：AI 寫的 pull request「frequently disregard code reuse opportunities」，審查者對它們卻表達了「more neutral or positive emotions」，結果是「the silent accumulation of technical debt」（[Huang 等人 2026](https://arxiv.org/abs/2601.21276)）；另一篇分析了 562 件被拒的 agentic PR，其中 142 件是重複提交，維護者直接指出已經有別人做了同一件事（[2026](https://arxiv.org/abs/2601.15195)）。

這種問題，人類其實見過。每一個塑膠袋、每一個塑膠瓶，都比它取代的東西更便宜、更好用；整體加起來，是淹沒海岸的垃圾。Labubu 那種盲盒公仔的熱潮更直接：刺激購買的機制本身，就在大量製造隨手就丟的塑膠。沒有人會說這是塑膠太有效率的錯。帳記在生產、消費和回收的機制上，記在設計這些機制、參與這些機制的人身上。

所以 AI 把這套評分機制玩爛，一點也不意外；該付責任的是濫用指標的人。這包括把解名題當成 benchmark、急著公布成績的 AI 公司，也包括只看榜單、不問結果有沒有被消化的圍觀群眾，以及用篇數和名次評價研究的機構。聲明最後一段其實也是這個意思：這些改變最後是好是壞，「will in large part be determined by the decisions of the humans in control of this new technology」。

那帳要怎麼付？Tao 給的答案是改計分：聲望應該轉向負責驗證和消化的人，社群應該停止把一個沒有消化過的原始證明當成完成的解答；消化要靠社群來認證，由期刊、課程、出版社訂出好寫作的標準。這正是聲明裡那個人類專家社群的位置：它不負責產出，負責篩選和消化。這個環節要運作，就得有人為它計分。

## 後記：一次回收

寫〈兩個 δ〉的過程，剛好是一次回收的實驗：把堆了七年的結果撿回來，消化成同一套說法。

那篇把 2019 年到 2025 年的後續研究一篇篇對回原文，把每一篇指認的旋鈕放進同一條一階變分的式子裡。寫這幾篇的四天裡，`references/` 累積了 147 份原始文獻，每一份都轉成純文字，用來逐條核對引文。七年沒有人做的歸位，這次花的是幾天。

Tao 說 AI 在越往後的環節加速越少，這是事實。這次被加速的是找文獻、對原文、查引句；哪一個旋鈕落在式子的哪一塊、哪一句話說過了頭，還是要人來判斷。但加速得少，不等於不能加速；更重要的是，把哪些工作交給 AI、交出去之後怎麼計分，仍然是人要付的帳，不該算在效率頭上。

這也不代表那次消化就是對的。它沒有經過審查。按 Tao 自己定的門檻，「if the authors cannot convincingly demonstrate that they can give a clear, expert-level talk on their results, that is correct and properly attributed, then the result should not be published」，清楚這一項或許做到了；正確與歸屬，還要經過審查。能被加速的是消化的成本；消化得對不對，還是要有人來看。

## 參考資料

- Fields 獎得主聯名 (2026). [A Severe Misalignment of AI in Mathematics](https://mathandai.org). 2026 年 9 月 11 日。
- Tao, T. (2026). [A Severe Misalignment of AI in Mathematics](https://terrytao.wordpress.com/2026/09/11/a-severe-misalignment-of-ai-in-mathematics/). 部落格，2026 年 9 月 11 日。
- Tao, T. [Terence Tao on AI — a living summary](https://teorth.github.io/tao-web/ai-views.html). 本文引用的三環節、Simpson's paradox 與出版門檻，出自這份整理。
- Nguyen, T., & Doğan, M. L. (2026). [Two responses to "A Severe Misalignment of AI in Mathematics"](https://proofsandprompts.com/2026/09/18/two-responses-to-a-severe-misalignment-of-ai-in-mathematics/). Proofs and Prompts，2026 年 9 月 18 日。
- Neal, R. M. (1995). [Bayesian Learning for Neural Networks](https://www.cs.toronto.edu/~radford/ftp/thesis.pdf). PhD thesis, University of Toronto.
- Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572v4). NeurIPS 2018.
- Yang, G., & Hu, E. J. (2021). [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522v3). ICML 2021.
- Huang, H., Jaisri, P., Shimizu, S., Chen, L., Nakashima, S., & Rodríguez-Pérez, G. (2026). [More Code, Less Reuse: Investigating Code Quality and Reviewer Sentiment towards AI-generated Pull Requests](https://arxiv.org/abs/2601.21276).
- (2026). [Where Do AI Coding Agents Fail? An Empirical Study of Failed Agentic Pull Requests in GitHub](https://arxiv.org/abs/2601.15195).
