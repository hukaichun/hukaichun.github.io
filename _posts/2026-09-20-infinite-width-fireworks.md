---
title: "Deep Learning 的無窮維煙花"
subtitle: "重讀 NTK：梯度只給方向，位移要乘上步長"
date: 2026-09-20
tags: [ntk, optimization, kernel-methods, math]
description: "無限寬網路等價於核方法，這個結論流行的版本是「夠寬就不必更新權重」。但梯度只給方向，位移要乘上步長，而 NTK 的設定把步長固定住了。剩下的那個對象，2007 年就有名字。"
---

前兩篇都在有限維裡打轉：[選項為什麼要一起讀](/posts/jev-is-not-new/)、[embedding 的維度夠不夠](/posts/decomposition-and-measure-zero/)。這篇看一個乾脆放手到無限維的設定。

Jacot、Gabriel 與 Hongler（[2018](https://arxiv.org/abs/1806.07572v4)）證明：寬度趨近無限時，神經正切核 $$\Theta$$ 在訓練中保持不變，網路的訓練動力學等價於用這個固定核做迴歸。流行的轉述是：**網路夠寬的話，權重根本不需要更新。**

這句話是誤讀，而且誤讀的位置很基本。

## 1. 梯度只給方向

以單隱藏層為例，採用 NTK parameterization：

$$
f(x) = \frac{1}{\sqrt{n}}\sum_{i=1}^{n} a_i\,\sigma(w_i^\top x)
$$

對隱藏層的梯度是

$$
\frac{\partial f}{\partial w_i} = \frac{1}{\sqrt{n}}\,a_i\,\sigma'(w_i^\top x)\,x
$$

寬度只改變這個量的大小，不改變它的方向。實際位移是步長乘上梯度，所以「權重動不動」取決於 $$\eta$$ 怎麼隨 $$n$$ 走——而 NTK 的設定把 $$\eta$$ 固定住了。

於是正確的敘述不是「夠寬就不用更新權重」，而是：**在這個參數化下、步長不隨寬度補償時，權重的位移趨近於零。**前者聽起來像是關於架構的發現，後者則看得出它是關於步長慣例的選擇。

修過最佳化的人都知道，梯度給的是方向，要談位移得先給步長。這個誤讀不深，卻紅了好一陣子。

## 2. 那個 $$1/\sqrt{n}$$ 不是隨便放的

要公平地看這件事，得先承認那個因子非放不可。

把縮放寫成 $$n^{-m}$$，初始化時輸出的變異數是

$$
\operatorname{Var}\big(f(x)\big) = n^{1-2m}\cdot O(1)
$$

$$m$$ 太大，輸出塌縮成零；太小，輸出隨寬度發散。只有 $$m = 1/2$$ 能讓輸出維持 $$O(1)$$，也就是讓初始化保有多樣性而不退化。這是被逼出來的，不是慣例。

但同一個因子也出現在梯度裡。維持前向傳播不退化的正規化，直接讓每個隱藏單元的梯度變成 $$O(n^{-1/2})$$。**為了讓初始化有意義而付的代價，是讓個別權重的更新消失。**

## 3. 權重不動，但函數照動

特徵學不動，那還能學什麼？

每個參數移動 $$O(\eta\,n^{-1/2})$$，每個參數對輸出的影響也是 $$O(n^{-1/2})$$，但這樣的項有 $$n$$ 個：

$$
\Delta f \;\approx\; \sum_{i}\frac{\partial f}{\partial \theta_i}\,\Delta\theta_i \;\sim\; n \times O(n^{-1/2}) \times O(\eta\,n^{-1/2}) \;=\; O(\eta)
$$

輸出的變化是 $$O(1)$$，資料擬合得起來。而特徵圖本身動不了：$$\varphi(x) = \sigma(W x)$$ 的相對變化是 $$\Theta(n^{-1/2})$$，$$\Theta$$ 本身的相對變化也是。

擬合照做，特徵不變——那被學到的就只剩讀出那一層：

> **固定的隨機特徵，加上一層線性讀出。**

$$\varphi(x) = \sigma(W_0 x)$$ 在初始化時抽好就凍結，訓練只調整 $$a$$。

## 4. 那把步長放大不就好了

自然的反應是：既然位移被 $$n^{-1/2}$$ 壓掉，把 $$\eta$$ 放大 $$\sqrt{n}$$ 倍補回來就是了。

不行，因為梯度下降的穩定性有上限。對二次目標，$$\eta > 2/\lambda_{\max}$$ 就會發散——這是最佳化課本的內容——而在這個正規化下 $$\Theta$$ 的最大特徵值是 $$O(1)$$，所以全域步長有一個跟寬度無關的天花板。

而這條線在實務上是緊的。Cohen 等人（[2021](https://arxiv.org/abs/2103.00065v3)）量到，訓練中 sharpness（損失 Hessian 的最大特徵值）會持續上升，直到碰到 $$2/\eta$$ 然後貼著它跑——他們稱為 edge of stability。你給多大的步長，系統就把自己推到那個步長允許的極限。

所以這不是「調個參數就好」的問題：在單一全域步長的框架裡，特徵要動就得付穩定性的代價。**特徵不動是全域步長這個慣例的後果。**

## 5. 這東西 2007 年就有名字

固定的隨機特徵加上線性讀出，不是一個新對象。

Johnson–Lindenstrauss 保證：$$N$$ 個點可以被線性映射到 $$O(\epsilon^{-2}\log N)$$ 維，兩兩距離的相對誤差不超過 $$\epsilon$$，而隨機投影以高機率就能辦到。Rahimi 與 Recht（[2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)）給出核的版本：對 shift-invariant 核，$$D = O(d\,\epsilon^{-2}\log\frac{1}{\epsilon^2})$$ 個隨機特徵就能在緊集上**均勻**逼近，這是逐點的保證，不是平均的。

先講清楚，這不是在貶低隨機投影。它很好用，而且是現在這套東西的基礎建設之一。最貼近的例子是 token 變成 vector：one-hot 是 $$\lvert V\rvert$$ 維的稀疏空間，embedding 矩陣就是把它壓到幾百維。初始化時，那字面上就是一個隨機投影，而 JL 保證不同的 token 不會被壓到撞在一起——取 $$\lvert V\rvert = 50{,}000$$、$$\epsilon = 0.5$$，JL 的界大約是 520 維，跟實務上常用的維度是同一個量級。訓練之後這個空間才長出語意，但它的起點確實是隨機投影，而且起點就已經夠用。

問題不在「退化成隨機投影」本身不好，而在於**這個對象的行為是已知的**。固定基底加線性讀出，它的收斂率由基底的譜、以及目標函數在那個譜上的展開決定——這是核方法統計學的古典內容，Caponnetto 與 De Vito（[2007](https://doi.org/10.1007/s10208-006-0196-8)）那一套，比 NTK 早了十年。基底沒對上目標，率就是會差；而「學特徵」的意思，正是把基底轉到目標上。

先記著這句話。後面那幾年的爭論，大半是在重新測量一個已經被刻畫過的對象。

## 6. 原文自己就是這樣寫的

特徵不會變不是後人的加工，是原文的賣點。

〈Neural Tangent Kernel〉的摘要寫著「in the infinite-width limit it converges to an explicit limiting kernel and **it stays constant during training**」，Theorem 2 前面標明這是論文的第二個主要結果。緊接在那句之後的是：「This makes it possible to study the training of ANNs in function space instead of parameter space.」

特徵凍結在原文裡是**好消息**，是讓分析變得可能的那個性質，不是一個需要被標註的限制。

邊界原文也畫了，只是沒被轉述。Theorem 2 是對**固定的時間區間** $$[0,T]$$ 敘述的：寬度趨近無限時，$$\Theta(t) \to \Theta_\infty$$ 在 $$t \in [0,T]$$ 上均勻成立。時間區間不能隨寬度一起長。

實驗設定更直白。他們寫道這個參數化「大幅減少連結權重在訓練中的影響」，而他們用的學習率是 1.0，比通常大，並說明這等價於「寬度 100 的傳統網路配上 0.01 的學習率」。**參數化與學習率是同一件事的兩種寫法，論文自己講了。**

把這件事講清楚的是 Chizat、Oyallon 與 Bach（[2019](https://arxiv.org/abs/1812.07956v5)）。引入顯式的尺度 $$\alpha$$ 之後，他們證明 lazy 現象「並非過參數化神經網路特有」，而是「一個常常是隱含的縮放選擇」造成的；只要模型在初始化時輸出接近零，**任何**可微模型都能被調成 lazy。「lazy training」這個名字就是他們取的，名字本身已經是一種糾正。

## 7. 後來的討論：離不開的，與離題的

之後幾年的文獻，可以用一個問題分類：把「lazy」換回「固定的隨機特徵加線性讀出」，那篇論文還剩下什麼。

**第一堆換完之後剩下同義反覆。**

Du 等人（[2019](https://arxiv.org/abs/1810.02054v2)）證明過參數化網路上的梯度下降收斂到全域最小值，關鍵假設是 Gram 矩陣的最小特徵值 $$\lambda_0 = \lambda_{\min}(H^\infty) > 0$$，而證明成立的原因正是訓練中 Gram 矩陣幾乎不變。換句話說：固定基底上的最小平方是凸的。

Arora 等人（[2019](https://arxiv.org/abs/1904.11955v2)）把對應的卷積核 CNTK 精確算出來。換句話說：他們設計了一個核。

Lee 等人（[2019](https://arxiv.org/abs/1902.06720v4)）證明寬網路的訓練動態就是線性化模型的動態。換句話說：線性模型是線性的。

三篇的定理都對，也都不是關於神經網路的結果，是關於固定基底加線性讀出的結果。它們之所以讀起來像發現，很大一部分是因為那個對象被叫做 neural tangent kernel，而不是 random features。

**第二堆換完之後離題。**

Chizat 等人主張 lazy 不太可能是深度學習成功的原因，招來的回應大多在量績效。Arora 等人的 CNTK 在 CIFAR-10 上有 77%，比其他核方法好一截；Geiger、Spigler、Jacot 與 Wyart（[2020](https://arxiv.org/abs/1906.08034v4)）掃過尺度與寬度，發現全連接網路在 lazy regime 下普遍表現較好。

把 lazy 換回定義，這兩句就變成：這個隨機投影是個不錯的隨機投影；以及，全連接網路本來就沒什麼特徵好學，隨機投影已經是它的天花板。兩句都是對的，兩句都不是反駁。

真正問對問題的只有一個方向：真網路是不是就待在這個 regime 裡。Lee 等人主張線性化模型與真實網路高度吻合，「即使對有限的、實務尺寸的網路」也成立，而且「跨架構、跨最佳化方法、跨損失函數都穩健」。如果成立，lazy regime 就不是人為挖出來的角落，而是真網路平常待的地方。

而兩邊自己的數據把界線畫在同一個位置。Geiger 等人的摘要第二點：「在我們的測試中，全連接網路在 lazy-training regime 下普遍表現較好，**卷積網路則不然**。」Lee 等人的討論節引 Novak 等人：無限寬在許多全連接與 locally-connected 架構上跟有限寬一樣好或更好，「然而在沒有 pooling 的卷積網路上，結論相反」。接著是一句老實話：「決定這些效能落差的主要因素，仍然是開放的研究問題。」

界線落在架構有沒有在做事。全連接層沒有結構可以利用，固定的隨機基底打平；卷積有，固定的隨機基底就輸。而這正是第 5 節那句話：基底沒對上目標，率就是會差。不是關於無窮寬的發現。

## 8. 還沒消化完

真正奇怪的是這條路線怎麼進步的。

Arora 等人怎麼從更差的版本推到 77%？換架構——有 global average pooling 的 CNTK 比 vanilla 版本高出 8% 到 9%。他們還提議把這套方法拿去做神經架構搜尋：先算核，測驗證集，用結果來挑架構。

把這串排在一起：

> 故事說「訓練不會改變 $$\Theta$$，這正是它可以被分析的原因」。
> 而在這條路線上要進步，唯一的辦法是**換個架構，手工換掉 $$\Theta$$**。

特徵不再由訓練學出來，改由人挑架構決定——這正是 2012 年以前的做法。而這套手工核設計，又被用來解釋深度學習為什麼比手工特徵強。

這條路線自己的數據也否證了那個故事。Arora 等人同時報告：CNTK 與對應的有限寬 CNN 之間仍有 5% 到 6% 的差距，並寫道這代表「有限寬度有它的好處」，因此「在 NTK regime 下的過參數化理論，目前還不能完全解釋神經網路的成功」。

而「我們在重新發明隨機投影」這句話，其實有人寫下來過。Chizat 等人在正文裡提了一句：兩層網路的線性化模型「是一個 random feature model，很適合拿來做統計分析」。

注意那個語氣——**這件事被當成方便，不是當成警訊。**同一個觀察可以用來說「所以我們終於可以分析它了」，也可以用來說「所以我們分析的不是神經網路」。這條路線選了前者，選了好幾年。

## 後記：證明可以對，故事可以錯

NTK 的定理是對的，推導可以逐行驗證。問題出在敘事層：它被說成「解釋了深度學習為什麼有效」，而它描述的是一個把特徵學習關掉的 regime。

這個落差的成因不深。梯度是方向不是位移，這是最佳化的第一課；而定理成立的條件寫在原文裡，連學習率等價於參數化這件事，Jacot 等人自己都註記了。

即便如此，社群還是需要另外一篇 NeurIPS 論文來把「這是縮放造成的」說清楚，並給它一個名字；之後又花了好幾年，才把「這個定理到底說了什麼、沒說什麼」消化乾淨。

糾正的成本遠高於犯錯的成本。而在文獻裡，一個正確的證明配上一個錯誤的故事，可以跑得比任何人預期的都遠。

這背後有一個結構性的誘因：可證明性本身就是篩選標準。一個把想理解的現象關掉、因而變得可以證明的設定，會拿到不成比例的注意力；而要糾正它，還得再寫一篇同樣可以被證明的論文。

還有一個更好量的版本：刷榜。

Arora 等人靠換架構把 CNTK 推高 8 到 9 個百分點，並提議拿這套流程去搜尋架構；Lee 等人（[2020](https://arxiv.org/abs/2007.15801v2)）那篇大規模實證，摘要的最後一句是「使用這些最佳實務，我們在所考慮的每一類架構對應的核上，取得了 CIFAR-10 分類的 state of the art」。

這些都是真的進步——**工程上的**。把一個已知的方法調得更好、把數字往上推幾個百分點，可比較、可累積，沒什麼好挑剔。

但它不產生科學。被調的那個對象在 2007 年就被刻畫完了；拿它去刷 2020 年的榜，得到的是一個更好的核，不是一個關於神經網路的認識。而這條路線宣稱要回答的問題是「深度學習為什麼有效」。

一條以理解為目標的路線，用工程的指標結了案。而換量尺的過程沒有人喊停——因為每一步都通過了檢驗。

這件事有人喊過，而且是從最高的講台喊的。

2017 年 NIPS 把 Test of Time 獎頒給〈Random Features for Large-Scale Kernel Machines〉——就是上面那篇 2007 年的論文。Ali Rahimi 上台領獎，然後說：

> Machine learning has become alchemy.

他要的東西很明確：「I would like to live in a society whose systems are built on top of verifiable, rigorous, thorough knowledge and not on alchemy.」（[講稿](https://www.youtube.com/watch?v=x7psGHgatGM)）

接下來四年，這個領域在他那篇論文上蓋出了大量的嚴謹：定理、證明、精確算出來的核、收斂率，每一條都對。只是換了個名字。

**所以嚴謹從來不是缺的那一塊。**NTK 這條路線最完整的部分正是它的嚴謹——證明沒問題，審稿沒問題，數字也沒問題。壞掉的是「這些證明到底在講什麼」。

寫這篇的同一個月，數學界正在對 AI 提出形狀相同的抱怨。9 月 11 日，26 位 Fields 獎得主聯名發表〈[A Severe Misalignment of AI in Mathematics](https://mathandai.org)〉，核心是一句話：

> Solving problems is only a tool and proxy for achieving the primary goal of conceptual understanding and insight.

他們擔心的是解答被急著公布，來不及好好寫下來、提煉方法、引用前人，於是「數學家之間那條關鍵的人類傳遞鏈」會斷掉。

注意他們要的不是嚴謹——數學家不缺嚴謹。他們要的是 conceptual understanding and insight，而且明講解題只是 proxy。**這個診斷比 2017 年那個準。**

但把立場設成「AI 對人」是搞錯對象了。NTK 這件事裡沒有 AI，從頭到尾都是人，而它照樣發生了。

問題在機器學習這個領域怎麼使用數學：理論多半是後設的——先有現象、先有數字，再補一套形式化上去；而那套形式化只要撐得住審稿就夠了，不必撐得住它聲稱描述的對象。[上一篇](/posts/decomposition-and-measure-zero/)那個 $$L^2$$ 收斂是同一種東西：定理完全正確，但 $$L^2$$ 的收斂對測度零的集合說不出任何話，而機器學習的資料集正是測度零。文字借來了，約束力沒有跟著借來。

所以真正的對照不是 AI 與人，是**數學家對上機器學習研究員**：兩群人面對形狀相同的毛病。差別不在有沒有人看出來——Rahimi 2017 年就在頒獎台上講了——而在那之後發生了什麼。他講完，這個領域接著花四年把他的論文重新做了一遍，全程沒有任何一步需要被擋下來，每一篇都過審了。數學家那邊是 26 個人一起署名。

把事情修回來的也是同一批人：有人回去讀原文，有人替現象重新命名，有人做實驗證明那個故事站不住。這件事慢，但它發生了。

所以 AI 改變的不是這個毛病，是它的流量。消化一個定理花了四年，那四年裡產出端還是人的速度。等到每週都冒出一個新證明，而消化端還是同一群人的時候，還會有人回頭嗎？

## 參考資料

- Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572v4). NeurIPS 2018.
- Chizat, L., Oyallon, E., & Bach, F. (2019). [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956v5). NeurIPS 2019.
- Du, S. S., Zhai, X., Póczos, B., & Singh, A. (2019). [Gradient Descent Provably Optimizes Over-parameterized Neural Networks](https://arxiv.org/abs/1810.02054v2). ICLR 2019.
- Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., & Wang, R. (2019). [On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/abs/1904.11955v2). NeurIPS 2019.
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720v4). NeurIPS 2019.
- Geiger, M., Spigler, S., Jacot, A., & Wyart, M. (2020). [Disentangling feature and lazy training in deep neural networks](https://arxiv.org/abs/1906.08034v4).
- Lee, J., Schoenholz, S. S., Pennington, J., Adlam, B., Xiao, L., Novak, R., & Sohl-Dickstein, J. (2020). [Finite Versus Infinite Neural Networks: an Empirical Study](https://arxiv.org/abs/2007.15801v2). NeurIPS 2020.
- Novak, R., Xiao, L., Lee, J., Bahri, Y., Yang, G., Hron, J., Abolafia, D. A., Pennington, J., & Sohl-Dickstein, J. (2019). Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes. ICLR 2019.
- Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065v3). ICLR 2021.
- Rahimi, A., & Recht, B. (2007). [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf). NIPS 2007.
- Rahimi, A. (2017). [NIPS Test of Time Award 演講](https://www.youtube.com/watch?v=x7psGHgatGM)。上述論文獲獎，演講主題為「Machine learning has become alchemy」。
- Caponnetto, A., & De Vito, E. (2007). [Optimal Rates for the Regularized Least-Squares Algorithm](https://doi.org/10.1007/s10208-006-0196-8). *Foundations of Computational Mathematics*, 7, 331–368.
- Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz Mappings into a Hilbert Space. *Contemporary Mathematics*, 26, 189–206.
- Fields 獎得主聯名 (2026). [A Severe Misalignment of AI in Mathematics](https://mathandai.org). 2026 年 9 月 11 日。
