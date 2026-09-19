---
title: "耦合項可以被分解，為什麼 embedding 還是不夠？"
subtitle: "如果 Jev 是 rerank，為什麼不能是 embedding"
date: 2026-09-20
tags: [retrieval, embedding, kernel-methods, math]
description: "定理說任何耦合項都能分解成兩邊的乘積，所以 cross-encoder 原則上就是無限維的 embedding。但定理只保證 L2 收斂，而我們在乎的那些點，測度恰好是零。Slepian 的集中問題給出有限項要成立的條件。"
---

[上一篇](/posts/jev-is-not-new/)的結論是：Jev 的運算形狀就是 rerank，選項到 runtime 才給，模型把整份清單一起讀進去打分。

那接下來的問題很自然：既然如此，為什麼不做成 embedding？把 state 編成一個向量，每個選項也編成一個向量，內積之後過 softmax 就結束了。這樣候選可以預先算好，成本差好幾個數量級。同樣的問題，檢索領域問了十幾年：為什麼不能只用向量相似度，非得在後面再接一個 cross-encoder？

這個問題不能用「embedding 做不到 runtime 選項」來打發，因為它做得到。CLIP 的 zero-shot 分類就是這樣運作的，類別在呼叫的當下才給，照樣能分類。後面第 6 節會把這些實務上的證據整理在一起。

而且從數學的角度看，情況對 embedding 更有利。有一個定理說：**任何把兩邊耦合在一起的函數，都可以被分解成兩邊各自的函數相乘再相加。**照這個定理，cross-encoder 原則上就是一個無限維的 embedding。兩者的差別只是維度多寡，不是種類不同。

但實務上沒有人真的這樣做，而且經驗一面倒：rerank 就是比 embedding 準，跨領域的時候差距還會拉大。如果定理說兩者只差在維度，那這個差距應該是量的問題；為什麼在實務上它看起來像質的問題？

這篇想把這個落差講清楚。路線是這樣：先把定理講準，弄明白它到底保證了什麼（第 1 節）；接著看它保證的是**哪一種收斂**，以及這種收斂對檢索來說為什麼是最糟的那一種（第 2 節）；然後用 Slepian 的集中問題，看有限項什麼時候才能給出我們真正要的那種保證，代價又是什麼（第 3 節）；最後把相關性函數放回這個座標系裡（第 4 節），談能怎麼處理（第 5 節），並用實驗和有限維的結果收尾（第 6 節）。

結論會落在一句話上：**定理保證的是平均意義下的收斂，而機器學習在乎的那些點——有限的資料集，尤其是其中決定排序的那條交界——測度恰好是零。**

## 1. 分解定理：embedding 在原則上是萬能的

給一個相關性函數 $$K(q, c)$$，它把 query 和候選耦合在一起。只要 $$K$$ 平方可積（Hilbert–Schmidt），Schmidt 分解就保證

$$
K(q, c) = \sum_{i=1}^{\infty} \sigma_i\, u_i(q)\, v_i(c)
$$

把 $$f(q) = (\sqrt{\sigma_i}\, u_i(q))_i$$、$$g(c) = (\sqrt{\sigma_i}\, v_i(c))_i$$ 各自看成一個向量，就有 $$K(q,c) = \langle f(q), g(c) \rangle$$。**任何 cross-encoder 在原則上都是一個無限維的 bi-encoder。**耦合項可以被拆開，這件事沒有商量餘地。

而且截斷還是最佳的：取前 $$d$$ 項得到的 $$K_d$$，在所有 rank 不超過 $$d$$ 的函數裡，$$L^2$$ 誤差最小，誤差平方就是被丟掉的 $$\sum_{i>d}\sigma_i^2$$（Schmidt 1907，也就是矩陣版的 Eckart–Young）。

順帶澄清一個常見的誤解：**不對稱不是障礙**。任何核都能拆成對稱與反對稱兩部分；更直接的說法是，把 $$K$$ 放進分塊矩陣

$$
\tilde K = \begin{pmatrix} 0 & K \\ K^\top & 0 \end{pmatrix}
$$

$$\tilde K$$ 本身就是對稱的，特徵值是 $$\pm\sigma_i$$，所以 $$K$$ 的 Schmidt 分解就是一個對稱核的特徵分解。對稱性是免費的。

所以問題不在能不能分解，而在**分解出來的級數是在什麼意義下收斂**。

## 2. 定理只保證 L² 收斂，而我們在乎的點測度是零

Schmidt 分解保證的是 $$L^2$$ 收斂，它蘊含依測度收斂，但不保證逐點收斂，更不保證均勻收斂。

要得到均勻而且絕對的收斂，得靠 Mercer 定理，而 Mercer 要求核是連續且**半正定**的。注意這裡卡住的是正定性，不是對稱性：上面那個 $$\tilde K$$ 雖然對稱，特徵值卻有正有負；反對稱的部分二次型恆為零，更不可能半正定。一般的相關性函數兩個條件都不滿足。

**而「只在 $$L^2$$ 意義下收斂」這種保證，對機器學習來說先天就有問題。**

理由很簡單：機器學習的資料集是有限的。有限集在任何非原子測度下的測度都是零。所以不管你的模型在整個空間上「平均而言」逼近得多好，這個保證對你手上那幾百萬筆資料點**一句話都沒說**。依測度收斂允許誤差集中在任何一個測度為零的集合上，而你的資料集正好就是這樣一個集合。

有人會說，那把測度換成資料的經驗測度不就好了？這確實是訓練在做的事，但這樣一來就換成另一隻角：經驗測度把質量全部放在已經看過的點上，於是保證只對那些點成立，對沒看過的配對完全沒有發言權。連續測度下資料是測度零，經驗測度下未見資料是測度零。兩隻角都躲不掉。

檢索則是這個一般問題的加強版。五萬份文件裡，每個 query 只有兩份相關，正例在所有配對裡佔的比例是 $$4 \times 10^{-5}$$。更糟的是，決定排序的是相關與不相關的**交界**，也就是相關性函數跳變的那條線，而它在整個 $$Q \times C$$ 上同樣是一個測度趨近於零的集合。

這不是抽象的擔憂，它有一個經典的名字：**Gibbs 現象**。方波的 Fourier 部分和在跳躍點附近會過衝，而且過衝不隨項數增加而消失，永遠停在跳躍幅度的 8.95% 左右。我實際算了一遍，$$n = 10$$ 時過衝 8.99%，$$n = 1000$$ 時還是 8.949%。項數增加一百倍，$$L^2$$ 誤差一路下降，跳躍點上的誤差卻紋風不動。

把同樣的事搬到相關性 kernel 上。我拿一個尖銳的 0/1 核（$$N = 1200$$，相關的定義是距離小於 0.05），截斷 $$d$$ 項之後分別量三個數字：

| 截斷 $$d$$ | $$L^2$$ 相對誤差 | 遠離交界的誤差 | 交界上的過衝 |
|---|---|---|---|
| 20 | 0.478 | 0.061 | 0.576 |
| 50 | 0.257 | 0.038 | 0.466 |
| 100 | 0.179 | 0.029 | 0.423 |
| 200 | 0.121 | 0.035 | 0.383 |
| 400 | 0.070 | 0.036 | 0.194 |

**遠離交界的地方逼近得很好，交界上始終是錯的。**而排序只由交界決定：一份文件相關與否，就是看它落在交界的哪一側。$$L^2$$ 誤差從 0.48 掉到 0.07，對我們真正在乎的那條線卻幾乎沒有幫助。

## 3. Slepian：有限項什麼時候才有均勻控制

那麼，有限項分解什麼時候才能給出每一個點都成立的控制？這正是 Slepian、Landau 與 Pollak 在 1960 年代問的問題。

他們問的是：一個訊號能不能同時集中在有限的時間區間和有限的頻帶裡？把這個問題寫成算子，核就是

$$
K(x, y) = \frac{\sin\big(c(x-y)\big)}{\pi(x-y)}
$$

它的特徵函數是 prolate spheroidal wave functions。真正的重點在特徵值的形狀：**它們不是慢慢衰減，而是一道階梯。**

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

Slepian 發現、Landau 與 Widom 嚴格證明了這個相變：以 $$c = \lvert T\rvert\,\lvert\Omega\rvert$$ 的正規化來說，$$\lambda_n \approx 1$$ 直到 $$n \approx c$$，之後迅速掉到 $$\approx 0$$，中間的墜落區寬度只有 $$\log c$$ 的量級。用我上圖的正規化（區間 $$[-1,1]$$、帶寬 $$c$$），轉折點在 $$2c/\pi$$，也就是 Shannon number。

我把這個數字實際算了一遍，$$\lambda_n > 0.5$$ 的個數是：

| $$c$$ | $$2c/\pi$$ | 實測個數 | 墜落區寬度 |
|---|---|---|---|
| 5 | 3.18 | 3 | 4 |
| 10 | 6.37 | 6 | 5 |
| 20 | 12.73 | 13 | 5 |
| 40 | 25.46 | 25 | 5 |

而且這個階梯帶來的好處正是前面缺的那一塊。截斷在墜落區之後，誤差是**均勻**小的：

| kernel | 截斷位置 | sup 誤差 |
|---|---|---|
| Slepian $$c = 20$$ | 13（Shannon number） | 0.39 |
| Slepian $$c = 20$$ | 18（墜落區之後） | $$1.9 \times 10^{-5}$$ |
| Slepian $$c = 20$$ | 23 | $$1.5 \times 10^{-11}$$ |
| 尖銳 0/1 | 200 | 0.56 |
| 尖銳 0/1 | 400 | 0.50 |

**十八維就讓每一個點的誤差小於十萬分之一，測度為零的點也不例外。**這不是「平均起來很好」，而是逐點的保證。

還有一個更深的連結。Landau 的定理可以用 Kolmogorov n-width 來表述：用 $$n$$ 維子空間逼近這整個函數類，最壞情況下的誤差就是 $$\sqrt{\lambda_n}$$。**特徵值的衰減速度，直接就是「最壞情況下 $$d$$ 維夠不夠」的答案。**階梯狀的譜，意思就是維度預算有一個誠實的數字：少了不行，多了沒用。

代價是什麼？是同時集中。Shannon number 是不確定性原理的定量版本：你想在兩個域同時集中，可以，但能用的自由度就是那麼多。這是一個具體的條件，不是「假設函數夠平滑」這種空話。

## 4. 相關性 kernel 在這個座標系裡的位置

現在可以把 embedding 的問題講清楚了。

Slepian kernel 是一個極端：它在兩個域都集中，所以譜是階梯狀的，有限維就真的夠，而且是均勻地夠。

檢索的相關性 kernel 在另一個極端。它在我們唯一在乎的地方——相關與不相關的交界——是尖銳的跳變。尖銳等於在對偶域展開得極寬，譜的尾巴又厚又長。上面那張圖裡，尖銳 kernel 那條線在 $$n = 59$$ 時還有 0.13，完全沒有階梯可言。

所以「embedding 要幾維才夠」這個問題，正確的問法是：**相關性 kernel 的譜有沒有階梯？**如果沒有，就不存在一個有限的 $$d$$，能在我們在乎的那些點上給出保證。而這件事跟模型多大、資料多乾淨都沒有關係，它是核本身的性質。

這也解釋了為什麼「換個更好的 encoder」救不了這件事。encoder 決定的是 $$u_i$$、$$v_i$$ 學得好不好，譜的形狀則是相關性本身決定的。

## 5. 那要怎麼處理在乎的那些點

有兩條路。

**第一條是不要預先分解。**cross-encoder 每次都直接算 $$K(q, c)$$ 這個值，不經過任何截斷，自然沒有截斷誤差的問題。代價是它不能預先計算，所以只能用在少量候選上。「先撈、再精排」這個工程慣例，其實就是這件事的實作：遠離交界的部分交給便宜的方法，交界附近交給逐點計算。

**第二條是換基底。**這正是 Slepian 函數本來的用途：不要用全域的基底，改用在你關心的區域上集中的基底。地球物理和天文就是這樣用的（[Simons, Dahlen & Wieczorek, 2006](https://arxiv.org/abs/math/0408424)），只分析南極或某個天區的時候，用在那個區域上最集中的那組函數，而不是整個球面的球諧函數。

檢索的對應物就是各種「把測度搬到重要配對上」的技巧：hard negative mining、領域內微調、針對困難 query 的專門訓練。這些做法的共同點，是承認全域的 $$L^2$$ 目標找錯了重點，於是手動把權重移到交界附近。

但這條路很拼。Slepian 函數之所以能成立，是因為「關心的區域」事先就知道，而且是一個固定的幾何區域。檢索不是這樣：相關與不相關的交界在哪裡，本來就是我們要找的答案。你得先知道答案，才能把基底集中在答案附近。

## 6. 實驗與有限維上的佐證

前面的論證都在無限維上進行。這一節收攏實務上的證據，順便看看同一件事在有限維怎麼呈現。

### 6.1 embedding 形式確實可行

先給 embedding 一個公道。它不是不能處理 runtime 才給的候選：CLIP（[Radford et al., 2021](https://arxiv.org/abs/2103.00020v1)）的 zero-shot 分類，就是把圖片編成一個向量、每個類別用「A photo of a {label}.」編成一個向量，內積後取 softmax，在 ImageNet 上拿到 76.2% 的準確率，而且完全沒有針對那 1,000 個類別訓練過分類頭。論文還提到類別向量算一次就能快取重複使用。

這正是 embedding 的價值所在，而且它的效率優勢是真的。Sentence-BERT（[Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084v1)）給過一個很具體的數字：在一萬個句子裡找出最相似的一對，用 BERT cross-encoder 需要大約五千萬次推論、約 65 小時，改用向量大約 5 秒。差距四萬多倍。

所以這篇不是在說 embedding 沒用，而是在問它的失效模式是什麼、為什麼那個失效模式沒辦法靠加維度解決。

### 6.2 有限維的版本：從 rank 到 margin

把 $$m$$ 個 query、$$n$$ 個候選的分數排成矩陣 $$S$$，bi-encoder 的分數矩陣是 $$S = F G^\top$$，所以 $$\operatorname{rank}(S) \le d$$。這是第 1 節那個分解在有限維的影子：維度就是保留的項數。

但檢索只在乎排序，所以要問的不是分數準不準，而是「符號對不對」。這個量叫 sign-rank：跟 0/1 相關性矩陣 $$A$$ 逐項同號的實矩陣裡，最小的 rank。它可以很大——隨機的正負號矩陣，sign-rank 與 $$n$$ 成線性（Alon、Frankl 與 Rödl, 1985）；Hadamard 矩陣至少 $$\sqrt{n}$$（[Forster, 2002](https://doi.org/10.1016/S0022-0000(02)00019-3)）。

不過單看 sign-rank 會低估 embedding 的難處，也會高估它。[Weller et al.（2025）](https://arxiv.org/abs/2508.21038v2)在論文第二版註明了一件有意思的事：「所有 $$k$$ 個一組的子集」這種結構，它的 sign-rank 只跟 $$k$$ 有關，跟 $$n$$ 無關。也就是說，**如果向量可以有無限精度，幾維就夠了。**

所以有限維的正確問法，跟前面幾節得到的答案一樣：要的是均勻的、帶間距的控制。他們的主定理就是這個版本——要讓所有 $$k$$ 子集都能以間距 $$\gamma$$ 被分開（向量取單位長度），必須有

$$
\binom{n}{k} \le \Big(1 + \frac{1}{\gamma}\Big)^d
$$

也就是 $$d \ge \log\binom{n}{k} / \log(1 + 1/\gamma)$$。論證是球面填充：你要能回答 $$\binom{n}{k}$$ 種「前 $$k$$ 名是誰」，每一種都需要球面上一個分得開的方向，而 $$d$$ 維球面在給定間距下塞得下的方向有限。**這裡的 $$\gamma$$ 就是本文一直在要的均勻控制，而不是平均誤差。**

取 $$\gamma = 0.1$$ 的話：一百萬份候選、$$k = 2$$ 時，下界只有 12 維，無害；$$k = 100$$ 時要約 425 維；$$k = 1000$$ 時要約 3,300 維。當 query 開始要求「同時滿足 A 且 B 但不是 C」這類組合，要表達的子集數量爆炸，預算很快就不夠。

他們還做了一個最理想的實驗：完全不用 encoder，直接在測試集上自由優化每個向量，看 $$k = 2$$ 時維度最多撐得住多少份候選。外插結果是 512 維約 50 萬份、1,024 維約 400 萬份、4,096 維約 2.5 億份。這是向量可以任意擺放的最好情況，真實的 encoder 只會更差。

### 6.3 LIMIT：把限制做成資料集

依照上面的理論，他們造了 LIMIT：五萬份文件、一千個 query，每個 query 恰好兩份相關文件，query 長得像「誰喜歡蘋果？」，文件長得像「Jon 喜歡蘋果」。

- 最強的單向量模型，recall@100 大多連 20% 都到不了。
- 縮到只有 46 份文件的版本，連 recall@20 都無法全對，最好的 recall@2 不到 60%。
- 拿 LIMIT 的訓練集去訓練幾乎沒用，所以問題不是領域不同，而是任務本身超出預算。
- 多向量的 GTE-ModernColBERT 明顯較好（小版本 recall@2 是 83.5%），但沒有解決。
- 把 Gemini-2.5-Pro 當 long-context reranker，一次給它 46 份文件和 1,000 個 query，全部答對。

還有一個漂亮的旁證：BM25 在小版本上拿到 97.8%。BM25 也是一種 embedding，只是維度是整個詞彙表，動輒數萬維。它能過關正是因為維度預算夠大——但把 query 換成同義詞之後，它掉了將近 90%。維度買到的是精度，不是語意。

### 6.4 測度一換，保證就沒了

BEIR（[Thakur et al., 2021](https://arxiv.org/abs/2104.08663v4)）在 18 個 zero-shot 資料集上比較各種檢索方法：BM25 加 cross-encoder rerank 整體最好，在 16 個資料集上贏過 BM25；dense retriever 效率高，卻常常落後。論文還指出，領域內的表現無法預測跨領域的表現——BM25 在 MS MARCO 上落後神經模型 7 到 18 分，換個領域卻非常強。

這正是第 2 節那個問題的實務版本：$$L^2$$ 的保證是相對於某一個測度的。訓練時的測度決定了哪些配對「重要」，encoder 就照那個重要性去分配它有限的維度預算。測度一換，分配就錯了，而錯的部分沒辦法在推論時補回來。cross-encoder 不需要事先分配，所以不受影響。

### 6.5 折衷方案也落在同一條軸上

ColBERT（[Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832v2)）保留每個 token 的向量，用 late interaction 計分，文件端仍可預先算好，但分數不再是單一內積。它比 BERT rerank 快兩個數量級、每個 query 的 FLOPs 少四個數量級，效果還能競爭；在 LIMIT 上則是「明顯較好但沒解決」。

用本文的語言說：它不是不做分解，而是把分解做得細一點——每個 token 一組座標，耦合延後到 MaxSim 那一步。延後得越多，越不受預算限制，成本也越高。整條光譜就是這一個取捨。

## 7. 結論

- 耦合項可以被分解，這是定理，沒有例外。所以 embedding 在原則上是萬能的。
- 但定理只保證 $$L^2$$ 收斂。機器學習的資料集是有限集，在連續測度下測度為零；依測度收斂對這種集合什麼都沒保證。檢索又是加強版：決定排序的是那條測度為零的交界。Gibbs 現象就是這件事的經典演示。
- 有限項要有均勻控制是可能的，Slepian 的集中問題給出了條件和代價：譜必須有階梯，而階梯來自同時集中。
- 相關性 kernel 沒有這個階梯，所以沒有一個有限的 $$d$$ 能在交界上給保證。
- 要處理那些點，要嘛逐點算，要嘛把基底搬過去。但後者需要事先知道答案在哪裡，而那正是我們要找的東西。
- 有限維的版本說的是同一件事：關鍵不是 sign-rank，而是帶間距的均勻分離；LIMIT 與 BEIR 則是這個限制在真實資料上的樣子。

## 附註：數值怎麼來的

文中的數字都是我自己算的，方法是把積分算子做 Nyström 離散化（$$N = 1200$$ 的均勻網格），再對離散矩陣取特徵值或奇異值。Gibbs 的過衝用方波的 Fourier 部分和直接計算，$$n = 1000$$ 時得到 8.9490%，與理論值相符。

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
