---
title: "耦合項可以被分解，為什麼 embedding 還是不夠？"
subtitle: "如果 Jev 是 rerank，為什麼不能是 embedding"
date: 2026-09-20
tags: [retrieval, embedding, kernel-methods, math]
description: "Schmidt 分解說 cross-encoder 就是無限維的 embedding，差別只在維度。但定理給的是 L2 收斂，而有限的資料集在連續測度下測度為零。Slepian 的集中問題給出有限項要有逐點控制的條件。"
---

[上一篇](/posts/jev-is-not-new/)的結論是：Jev 的運算形狀就是 rerank，選項到 runtime 才給，整份清單一起讀進去打分。那為什麼不做成 embedding？兩邊各編一個向量、內積、softmax，候選還能預先算好，成本差好幾個數量級。同樣的問題檢索領域問了十幾年：為什麼非得在向量檢索後面再接一個 cross-encoder？

數學上，這個問題的答案看起來是肯定的。只要相關性函數 $$K(q,c)$$ 平方可積，Schmidt 分解就給出

$$
K(q, c) = \sum_{i=1}^{\infty} \sigma_i\, u_i(q)\, v_i(c)
$$

取 $$f(q) = (\sqrt{\sigma_i}\,u_i(q))_i$$、$$g(c) = (\sqrt{\sigma_i}\,v_i(c))_i$$，就有 $$K = \langle f, g\rangle$$。**cross-encoder 就是一個無限維的 bi-encoder**，兩者的差別只是維度，不是種類。截斷前 $$d$$ 項還是最優的：在所有 rank $$\le d$$ 的函數裡，$$L^2$$ 誤差最小，誤差平方是 $$\sum_{i>d}\sigma_i^2$$。

實務上的經驗卻一面倒：rerank 比 embedding 準，換個領域差距還會拉大。如果只差在維度，這應該是量的問題。為什麼看起來像質的問題？

差別在**收斂的模式**。Schmidt 分解給的是 $$L^2$$ 收斂，而這種收斂允許誤差集中在任何一個測度為零的集合上。接下來的問題就是：$$L^2$$ 這種保證，對機器學習到底夠不夠？

## 1. 有限的資料集就是測度零

不夠，而且理由不特別針對檢索。

機器學習的資料集是有限集，在任何非原子測度下測度為零。依測度收斂允許誤差集中在測度為零的集合上，而你手上那批資料正好是這種集合。換成經驗測度也不解決問題，只是把洞換一個位置：保證跟著原子跑，未見的配對變成測度零。

檢索是這個問題的加強版。LIMIT 那種設定裡，五萬份文件、每個 query 兩份相關，正例佔所有配對的 $$4 \times 10^{-5}$$；而真正決定排序的還不是正例本身，是相關與不相關的**交界**，也就是 $$K$$ 跳變的那條線。

而在這條線上，截斷**不會**收斂，這是可以直接說死的。

若 $$u_i$$、$$v_i$$ 連續，則每個 $$K_d$$ 都連續；連續函數的均勻極限必連續。所以只要 $$K$$ 在交界上有跳躍 $$J$$，$$\lVert K - K_d \rVert_\infty \to 0$$ 就不可能成立，而且對任何 $$d$$ 都有

$$
\lVert K - K_d \rVert_\infty \ge \tfrac{J}{2}
$$

部分和在跳躍點收斂到左右極限的中點，那一點的誤差就是 $$J/2$$。Gibbs 現象說的是鄰域還更糟：過衝穩定在跳躍幅度的 8.95% 左右（Wilbraham–Gibbs 常數 $$\tfrac{2}{\pi}\int_0^\pi \tfrac{\sin t}{t}\,dt = 1.17898\ldots$$），於是 sup 誤差大約是 $$0.59J$$。兩個數字都與 $$d$$ 無關。

$$L^2$$ 誤差一路下降的同時，交界上的誤差完全不動。而排序只由交界決定。

## 2. 那有限項什麼時候才有逐點控制？

這才是該問的問題。截斷要有意義，需要的不是 $$\sum_{i>d}\sigma_i^2$$ 小，而是尾巴在**每一點**上都小。什麼樣的核做得到？

條件可以直接從結構讀出來：尾巴要能被均勻控制，譜就不能只是慢慢衰減，而要在某個位置斷掉。也就是說，$$\sigma_i$$ 應該像階梯——前面一段接近常數，然後迅速掉到可以忽略。這種核存在嗎？存在，而且它不是為了機器學習發明的。

1960 年代，Slepian、Landau 與 Pollak 問了一個通訊上的問題：一個訊號能不能同時集中在有限的時間區間和有限的頻帶裡？把這個問題寫成算子，核是

$$
K(x, y) = \frac{\sin\big(c(x-y)\big)}{\pi(x-y)}
$$

特徵函數是 prolate spheroidal wave functions，而特徵值正是那道階梯。

<figure>
<svg class="viz" viewBox="0 0 720 380" width="100%" role="img" aria-labelledby="specTitle specDesc" xmlns="http://www.w3.org/2000/svg">
<title id="specTitle">兩種 kernel 的譜</title>
<desc id="specDesc">Slepian kernel 的特徵值在 Shannon number 附近由 1 陡降到 0，墜落區只有幾個特徵值寬；尖銳 0/1 kernel 的奇異值緩慢衰減，沒有階梯。</desc>
<line class="grid" x1="52" y1="334.0" x2="555" y2="334.0"/>
<text class="tick" x="42" y="338.0" text-anchor="end">0</text>
<line class="grid" x1="52" y1="180.0" x2="555" y2="180.0"/>
<text class="tick" x="42" y="184.0" text-anchor="end">0.5</text>
<line class="grid" x1="52" y1="26.0" x2="555" y2="26.0"/>
<text class="tick" x="42" y="30.0" text-anchor="end">1</text>
<text class="tick" x="52.0" y="354" text-anchor="middle">0</text>
<text class="tick" x="137.3" y="354" text-anchor="middle">10</text>
<text class="tick" x="222.5" y="354" text-anchor="middle">20</text>
<text class="tick" x="307.8" y="354" text-anchor="middle">30</text>
<text class="tick" x="393.0" y="354" text-anchor="middle">40</text>
<text class="tick" x="478.3" y="354" text-anchor="middle">50</text>
<line class="mark" x1="160.5" y1="26" x2="160.5" y2="334.0"/>
<text class="marklab" x="154.5" y="18" text-anchor="end">2c/π ≈ 12.7</text>
<line class="mark" x1="269.1" y1="26" x2="269.1" y2="334.0"/>
<text class="marklab" x="275.1" y="18" text-anchor="start">2c/π ≈ 25.5</text>
<path class="s3" d="M 52.0 26.00 L 60.5 27.75 L 69.1 29.00 L 77.6 31.25 L 86.1 33.77 L 94.6 37.17 L 103.2 40.88 L 111.7 45.38 L 120.2 50.21 L 128.7 55.76 L 137.3 61.62 L 145.8 68.13 L 154.3 74.93 L 162.8 82.33 L 171.4 90.00 L 179.9 98.07 L 188.4 106.63 L 196.9 115.42 L 205.5 124.42 L 214.0 133.81 L 222.5 143.32 L 231.0 153.12 L 239.6 163.00 L 248.1 173.10 L 256.6 183.20 L 265.1 193.43 L 273.7 203.61 L 282.2 213.86 L 290.7 224.01 L 299.2 234.16 L 307.8 244.14 L 316.3 254.04 L 324.8 263.68 L 333.3 267.01 L 341.9 267.25 L 350.4 267.81 L 358.9 267.92 L 367.4 268.66 L 376.0 268.88 L 384.5 270.09 L 393.0 270.21 L 401.5 271.69 L 410.1 272.01 L 418.6 273.16 L 427.1 273.88 L 435.6 274.05 L 444.2 276.33 L 452.7 276.42 L 461.2 279.06 L 469.7 279.29 L 478.3 282.10 L 486.8 282.37 L 495.3 282.43 L 503.8 285.47 L 512.4 285.87 L 520.9 289.11 L 529.4 289.57 L 537.9 291.27 L 546.5 293.13 L 555.0 293.43" fill="none"/>
<line class="s3 swatch" x1="563" y1="293.4" x2="579" y2="293.4"/>
<text class="lab" x="585" y="297.4">尖銳 0/1 kernel</text>
<path class="s1" d="M 52.0 26.00 L 60.5 26.00 L 69.1 26.00 L 77.6 26.00 L 86.1 26.00 L 94.6 26.00 L 103.2 26.00 L 111.7 26.01 L 120.2 26.08 L 128.7 26.79 L 137.3 32.43 L 145.8 63.05 L 154.3 152.65 L 162.8 263.47 L 171.4 318.53 L 179.9 331.71 L 188.4 333.74 L 196.9 333.97 L 205.5 334.00 L 214.0 334.00 L 222.5 334.00 L 231.0 334.00 L 239.6 334.00 L 248.1 334.00 L 256.6 334.00 L 265.1 334.00 L 273.7 334.00 L 282.2 334.00 L 290.7 334.00 L 299.2 334.00 L 307.8 334.00 L 316.3 334.00 L 324.8 334.00 L 333.3 334.00 L 341.9 334.00 L 350.4 334.00 L 358.9 334.00 L 367.4 334.00 L 376.0 334.00 L 384.5 334.00 L 393.0 334.00 L 401.5 334.00 L 410.1 334.00 L 418.6 334.00 L 427.1 334.00 L 435.6 334.00 L 444.2 334.00 L 452.7 334.00 L 461.2 334.00 L 469.7 334.00 L 478.3 334.00 L 486.8 334.00 L 495.3 334.00 L 503.8 334.00 L 512.4 334.00 L 520.9 334.00 L 529.4 334.00 L 537.9 334.00 L 546.5 334.00 L 555.0 334.00" fill="none"/>
<line class="s1 swatch" x1="563" y1="314.0" x2="579" y2="314.0"/>
<text class="lab" x="585" y="318.0">Slepian c=20</text>
<path class="s2" d="M 52.0 26.00 L 60.5 26.00 L 69.1 26.00 L 77.6 26.00 L 86.1 26.00 L 94.6 26.00 L 103.2 26.00 L 111.7 26.00 L 120.2 26.00 L 128.7 26.00 L 137.3 26.00 L 145.8 26.00 L 154.3 26.00 L 162.8 26.00 L 171.4 26.00 L 179.9 26.00 L 188.4 26.00 L 196.9 26.00 L 205.5 26.00 L 214.0 26.01 L 222.5 26.06 L 231.0 26.44 L 239.6 28.88 L 248.1 41.58 L 256.6 88.10 L 265.1 183.91 L 273.7 276.95 L 282.2 319.97 L 290.7 331.41 L 299.2 333.60 L 307.8 333.95 L 316.3 333.99 L 324.8 334.00 L 333.3 334.00 L 341.9 334.00 L 350.4 334.00 L 358.9 334.00 L 367.4 334.00 L 376.0 334.00 L 384.5 334.00 L 393.0 334.00 L 401.5 334.00 L 410.1 334.00 L 418.6 334.00 L 427.1 334.00 L 435.6 334.00 L 444.2 334.00 L 452.7 334.00 L 461.2 334.00 L 469.7 334.00 L 478.3 334.00 L 486.8 334.00 L 495.3 334.00 L 503.8 334.00 L 512.4 334.00 L 520.9 334.00 L 529.4 334.00 L 537.9 334.00 L 546.5 334.00 L 555.0 334.00" fill="none"/>
<line class="s2 swatch" x1="563" y1="346.0" x2="579" y2="346.0"/>
<text class="lab" x="585" y="350.0">Slepian c=40</text>
<text class="axis" x="304" y="374" text-anchor="middle">序號 n</text>
<text class="axis" transform="translate(14,180) rotate(-90)" text-anchor="middle">大小（正規化）</text>
</svg>
<figcaption>兩種 kernel 的譜。Slepian kernel 的特徵值在 Shannon number 附近由 1 陡降到 0；尖銳的 0/1 kernel 只是緩慢衰減，沒有階梯。</figcaption>
</figure>

Slepian 發現、Landau 與 Widom 證明了這個相變：以 $$c = \lvert T\rvert\,\lvert\Omega\rvert$$ 正規化時，$$\lambda_n \approx 1$$ 直到 $$n \approx c$$，之後迅速掉到 $$\approx 0$$，墜落區寬度只有 $$\log c$$ 的量級。上圖用的正規化是區間 $$[-1,1]$$、帶寬 $$c$$，轉折點落在 $$2c/\pi$$，也就是 Shannon number。

這張圖真正在說的，是**截斷對目標定義域的作用**。保留 $$d$$ 項，等於把目標換成它在前 $$d$$ 個模態上的投影：你不再逼近原本那個 $$K$$，而是逼近一個被縮小過的目標。差別在於這件事要付多少代價。

譜是階梯的時候，在墜落之後截斷，被丟掉的部分本來就微不足道——縮小後的定義域幾乎就是原來的定義域，所以逐點的控制拿得到，而 Shannon number 就是這個定義域誠實的維度。Landau 的定理用 Kolmogorov n-width 說的是同一件事：用 $$n$$ 維子空間逼近整個函數類，最壞情況的誤差是 $$\sqrt{\lambda_n}$$。譜的衰減就是「最壞情況下 $$d$$ 維夠不夠」的答案。

譜是長尾的時候，截斷同樣縮小了定義域，但縮掉的正是交界附近那些高頻的部分——也就是唯一決定排序的東西。差別不在逼近得好不好，而在你其實已經換了一個問題。

代價寫在 Shannon number 裡：$$2c/\pi$$ 是不確定性原理的定量版本。要有這種保證，核必須在兩個域同時集中，而能用的自由度就是那麼多。

## 3. 相關性核在這個座標系的位置

Slepian 核是一個極端：兩個域都集中，譜是階梯，有限維真的夠。

相關性核在另一個極端。它在交界上是跳變的，對偶域展得極寬，譜的尾巴又厚又長——上圖那條線在 $$n = 59$$ 時還有 0.13。「embedding 要幾維」這個問題因此有了明確的問法：**相關性核的譜有沒有階梯？**沒有的話，就不存在一個有限的 $$d$$，能在交界上給出保證。這跟模型多大、資料多乾淨無關，是核本身的性質；換更好的 encoder 只改變 $$u_i$$、$$v_i$$ 學得多好，改變不了 $$\sigma_i$$ 的形狀。

剩下的選擇只有兩個。

**不預先分解。**cross-encoder 每次直接算 $$K(q,c)$$，沒有截斷就沒有截斷誤差，代價是不能預先計算。「先撈、再精排」就是這件事的工程版本：遠離交界的部分交給便宜的方法，交界附近逐點算。

**換基底。**Slepian 函數本來就是為此而生：不用全域基底，改用在關心的區域上集中的那一組。地球物理和天文這樣用（[Simons, Dahlen & Wieczorek, 2006](https://arxiv.org/abs/math/0408424)），只看南極的時候就用在南極最集中的那組函數。檢索的對應物是 hard negative mining、領域內微調這類把測度搬到重要配對上的做法。

但第二條路有個結構性的問題：Slepian 函數能成立，是因為關心的區域事先已知，而且是固定的幾何區域。檢索的交界在哪裡，本身就是要找的答案。你得先知道答案，才能把基底集中到答案附近。

## 4. 有限維與實驗

前面都在無限維上談。同一件事在有限維的樣子，以及實驗證據。

**維度、sign-rank 與間距。**bi-encoder 的分數矩陣 $$S = FG^\top$$ 滿足 $$\operatorname{rank}(S) \le d$$；只在乎排序時，界是 sign-rank，它可以很大（隨機正負號矩陣與 $$n$$ 成線性，Alon–Frankl–Rödl 1985；Hadamard 至少 $$\sqrt{n}$$，[Forster 2002](https://doi.org/10.1016/S0022-0000(02)00019-3)）。但 [Weller et al.（2025）](https://arxiv.org/abs/2508.21038v2)在第二版註明：「所有 $$k$$ 子集」這種結構的 sign-rank 只跟 $$k$$ 有關。無限精度下幾維就夠——這正是第 2 節那個分野在有限維的翻版，光看 $$L^2$$ 或光看符號都會低估問題。

他們的主定理因此改用間距 $$\gamma$$：單位向量、所有 $$k$$ 子集都要以 $$\gamma$$ 分開，則 $$\binom{n}{k} \le (1 + 1/\gamma)^d$$，由球面填充得出。取 $$\gamma = 0.1$$、一百萬份候選：$$k = 2$$ 只要 12 維，$$k = 100$$ 要約 425 維，$$k = 1000$$ 要約 3,300 維。他們另外直接在測試集上自由優化向量（不經 encoder），外插得到 512 維約撐 50 萬份、1,024 維約 400 萬份、4,096 維約 2.5 億份。

**LIMIT。**五萬份文件、一千個 query、每個 query 兩份相關，題目長得像「誰喜歡蘋果」。最強的單向量模型 recall@100 大多不到 20%；46 份文件的小版本連 recall@20 都無法全對。拿訓練集去訓練沒用，所以不是領域差異。多向量的 GTE-ModernColBERT 好一些（小版本 recall@2 是 83.5%）但沒解決。把 Gemini-2.5-Pro 當 long-context reranker，46 份文件、1,000 個 query 一次餵進去，全部答對。附帶一提，BM25 在小版本拿到 97.8%——它也是 embedding，只是維度是整個詞彙表；但換成同義詞就掉了近 90%，維度買到的是精度，不是語意。

**BEIR。**18 個 zero-shot 資料集裡，BM25 加 cross-encoder rerank 整體最好，16 個資料集贏過 BM25；dense retriever 常落後，而且領域內的表現無法預測跨領域表現。這是測度換掉的後果：訓練時的測度決定 encoder 怎麼分配有限的維度預算，測度一換，分配就錯了，推論時補不回來。

**ColBERT。**保留 token 級向量、用 MaxSim 延後耦合，文件端仍可預算，比 BERT rerank 快兩個數量級，在 LIMIT 上「明顯較好但沒解決」。耦合延後得越多，越不受預算限制，成本也越高。整條光譜就是這一個取捨。

## 5. 結論

Schmidt 分解說 cross-encoder 就是無限維的 embedding，這是對的，但它保證的是 $$L^2$$ 收斂；而機器學習的資料集有限，在連續測度下測度為零，決定排序的交界更是。在交界上，截斷根本不收斂：連續的部分和不可能均勻逼近一個有跳躍的函數，誤差有一個與 $$d$$ 無關的下界。有限項要在這些點上有保證，譜必須有階梯，而 Slepian 告訴我們階梯的代價是兩個域同時集中。相關性核不具備這個性質，所以差距不是維度不夠，是保證的種類不對。

（圖中的譜是用 $$N = 1200$$ 的均勻網格做 Nyström 離散化後算出來的，只作示意；理論上的敘述不依賴這個數值。）

[下一篇](/posts/infinite-width-fireworks/)把維度推到無限：如果有限維的預算是問題，那無限寬的網路呢？

## 參考資料

- Slepian, D., & Pollak, H. O. (1961). Prolate Spheroidal Wave Functions, Fourier Analysis and Uncertainty — I. *Bell System Technical Journal*, 40(1), 43–63.
- Landau, H. J., & Pollak, H. O. (1961, 1962). Prolate Spheroidal Wave Functions, Fourier Analysis and Uncertainty — II, III. *BSTJ*, 40(1), 65–84; 41(4), 1295–1336.
- Landau, H. J., & Widom, H. (1980). Eigenvalue Distribution of Time and Frequency Limiting. *J. Math. Anal. Appl.*, 77(2), 469–481.
- Franceschetti, M. (2015). [On Landau's Eigenvalue Theorem and Information Cut-Sets](https://doi.org/10.1109/TIT.2015.2456878). *IEEE Trans. Inf. Theory*, 61(9).
- Kulikov, A. (2023). [Exponential Lower Bound for the Eigenvalues of the Time-Frequency Localization Operator before the Plunge Region](https://arxiv.org/abs/2306.12430).
- Simons, F. J., Dahlen, F. A., & Wieczorek, M. A. (2006). [Spatiospectral Concentration on a Sphere](https://arxiv.org/abs/math/0408424). *SIAM Review*, 48(3), 504–536.
- Hewitt, E., & Hewitt, R. E. (1979). The Gibbs–Wilbraham Phenomenon: An Episode in Fourier Analysis. *Archive for History of Exact Sciences*, 21(2), 129–160.
- Weller, O., Boratko, M., Naim, I., & Lee, J. (2025). [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038v2). ICLR 2026.
- Alon, N., Frankl, P., & Rödl, V. (1985). Geometrical Realization of Set Systems and Probabilistic Communication Complexity. *FOCS 1985*, 277–280.
- Forster, J. (2002). [A Linear Lower Bound on the Unbounded Error Probabilistic Communication Complexity](https://doi.org/10.1016/S0022-0000(02)00019-3). *JCSS*, 65(4), 612–625.
- Radford, A., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020v1).
- Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084v1).
- Khattab, O., & Zaharia, M. (2020). [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832v2).
- Thakur, N., et al. (2021). [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663v4).
