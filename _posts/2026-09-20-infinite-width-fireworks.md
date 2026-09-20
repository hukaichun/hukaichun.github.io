---
title: "Deep Learning 的無窮維煙花"
subtitle: "重讀 NTK：梯度只給方向，位移要乘上步長"
date: 2026-09-20
tags: [ntk, optimization, kernel-methods, math]
description: "無限寬網路等價於核方法，這個結論流行的版本是「夠寬就不必更新權重」。但梯度只給方向，位移要乘上步長，而 NTK 的設定把步長固定住了。後續這條路線的進展方式，反過來否證了它自己的故事。"
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

這裡有個轉折，講漏了就會把整件事理解錯。

權重幾乎不動，不代表模型學不動。每個參數移動 $$O(\eta\,n^{-1/2})$$，每個參數對輸出的影響也是 $$O(n^{-1/2})$$，但這樣的項有 $$n$$ 個：

$$
\Delta f \;\approx\; \sum_{i}\frac{\partial f}{\partial \theta_i}\,\Delta\theta_i \;\sim\; n \times O(n^{-1/2}) \times O(\eta\,n^{-1/2}) \;=\; O(\eta)
$$

輸出的變化是 $$O(1)$$，資料照樣擬合得起來。動不了的是特徵圖：$$\varphi(x) = \sigma(W x)$$ 的相對變化是 $$\Theta(n^{-1/2})$$，$$\Theta$$ 本身的相對變化也是。

所以這個 regime 的內容其實可以一句話說完：

> **固定的隨機特徵，加上一層線性讀出。**

$$\varphi(x) = \sigma(W_0 x)$$ 在初始化時抽好就凍結，訓練只調整 $$a$$。這不是神經網路的訓練，是隨機投影加線性迴歸——Rahimi 與 Recht（[2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)）那條 random features 路線的標準寫法。

## 4. 定理自己就畫了邊界

值得強調的是，這些限制多半寫在原文裡，只是沒被轉述。

Jacot 等人的 Theorem 2 是**連續時間**的敘述：參數跟隨一條 ODE，結論是對**固定的時間區間** $$[0, T]$$，當寬度趨近無限時 $$\Theta(t) \to \Theta_\infty$$ 在 $$t \in [0,T]$$ 上均勻成立。時間區間不能隨寬度一起長。

原文的實驗設定更直白。他們寫道，這個參數化「大幅減少連結權重在訓練中的影響」，而他們用的學習率是 1.0，比通常大，並說明這等價於「寬度 100 的傳統網路配上 0.01 的學習率」。**參數化與學習率是同一件事的兩種寫法，論文自己講了。**

Chizat、Oyallon 與 Bach（[2019](https://arxiv.org/abs/1812.07956v5)）把這件事推到底：引入顯式的尺度 $$\alpha$$ 之後，他們證明 lazy 現象「並非過參數化神經網路特有」，而是「一個常常是隱含的縮放選擇」造成的；只要模型在初始化時輸出接近零，**任何**可微模型都能被調成 lazy。他們也在 CIFAR-10 上用 VGG-11 實測，隨著 $$\alpha$$ 變大表現變差，並據此主張 lazy training 不太可能是神經網路成功的原因。「lazy training」這個名字就是他們取的，名字本身已經是一種糾正。

## 5. 那把步長放大不就好了

自然的反應是：既然位移被 $$n^{-1/2}$$ 壓掉，把 $$\eta$$ 放大 $$\sqrt{n}$$ 倍補回來就是了。

不行，因為梯度下降的穩定性有上限。對二次目標，$$\eta > 2/\lambda_{\max}$$ 就會發散，而在這個正規化下 $$\Theta$$ 的最大特徵值是 $$O(1)$$，所以全域步長有一個跟寬度無關的天花板。

這不只是紙上的限制。Cohen 等人（[2021](https://arxiv.org/abs/2103.00065v3)）觀察到，實際訓練中 sharpness（損失 Hessian 的最大特徵值）會持續上升，直到碰到 $$2/\eta$$ 這條線，然後就貼著它跑——他們稱為 edge of stability。你給多大的步長，系統就把自己推到那個步長允許的極限。

所以這不是「調個參數就好」的問題：在單一全域步長的框架裡，特徵要動就得付穩定性的代價。這也讓前面那個結論更清楚——**特徵不動是全域步長這個慣例的後果**。

## 6. 彎路

有意思的在後面。如果故事是「$$\Theta$$ 不會變，所以我們可以用它解釋深度學習」，那這條路線上要怎麼進步？

Du 等人（[2019](https://arxiv.org/abs/1810.02054v2)）證明了過參數化網路上的梯度下降會收斂到全域最小值，關鍵假設是 Gram 矩陣的最小特徵值 $$\lambda_0 = \lambda_{\min}(H^\infty) > 0$$，而證明之所以成立，正是因為訓練中 Gram 矩陣幾乎不變。**這個收斂定理的前提，就是不學特徵。**

接著 Arora 等人（[2019](https://arxiv.org/abs/1904.11955v2)）把對應的卷積核 CNTK 精確算出來，在 CIFAR-10 上得到 77%。怎麼從更差的版本進步到這個數字？換架構：有 global average pooling 的 CNTK 比 vanilla 版本高出 8% 到 9%。他們還提議把這套方法拿去做神經架構搜尋——先算核，測驗證集，用結果來挑架構。

把這串排在一起，脈絡就很奇怪：

> 故事說「訓練不會改變 $$\Theta$$，這正是它可以被分析的原因」。
> 而在這條路線上要進步，唯一的辦法是**換個架構，手工換掉 $$\Theta$$**。

特徵不再由訓練學出來，改由人挑架構決定——這正是 2012 年以前的做法。深度學習最核心的那件事被關掉之後，剩下的工作就是核設計，而這套核設計又被用來解釋深度學習為什麼成功。

而且這條路線自己的數據就否證了那個故事。Arora 等人同時報告：CNTK 與對應的有限寬 CNN 之間仍有 5% 到 6% 的差距，並寫道這代表「有限寬度有它的好處」，因此「在 NTK regime 下的過參數化理論，目前還不能完全解釋神經網路的成功」。

那 5% 到 6% 就是特徵學習的價值。

要說公道話：這些論文本身是誠實的。Arora 等人明白寫出限制，把 CNTK 當成工具而不是解釋；Du 等人的定理在自己的假設下完全正確。反智的不是論文，是中間那層轉述——「無限寬解釋了深度學習」——而那句話並不在論文裡。

## 7. 替隨機投影說句話

上面一路在說「退化成隨機投影」，但這不是在貶低隨機投影。它是很好的東西，只是不能拿來當深度學習的解釋。

Johnson–Lindenstrauss 保證：$$N$$ 個點可以被線性映射到 $$O(\epsilon^{-2}\log N)$$ 維，兩兩距離的相對誤差不超過 $$\epsilon$$，而隨機投影以高機率就能辦到。Rahimi 與 Recht 的 random features 則給出核的版本：對 shift-invariant 核，$$D = O(d\,\epsilon^{-2}\log\frac{1}{\epsilon^2})$$ 個隨機特徵就能在緊集上**均勻**逼近，這是逐點的保證，不是平均的。

更貼近現在的例子是 token 變成 vector。one-hot 是 $$\lvert V\rvert$$ 維的稀疏空間，embedding 矩陣就是把它壓到幾百維。初始化時，那字面上就是一個隨機投影，而 JL 保證不同的 token 不會被壓到撞在一起：取 $$\lvert V\rvert = 50{,}000$$、$$\epsilon = 0.5$$，JL 的界大約是 520 維——跟實務上常用的維度是同一個量級。訓練之後這個空間才長出語意，但它的起點確實是隨機投影，而且起點就已經夠用。

所以隨機投影不但沒錯，還是現在這套東西的基礎建設之一。NTK 的問題從來不是「退化成隨機投影」這件事本身不好，而是**隨機投影解釋不了為什麼深度學習比隨機投影強**。

## 後記：證明可以對，故事可以錯

NTK 的定理是對的，推導可以逐行驗證。問題出在敘事層：它被說成「解釋了深度學習為什麼有效」，而它描述的是一個把特徵學習關掉的 regime。

這個落差的成因不深。梯度是方向不是位移，這是最佳化的第一課；而定理成立的條件寫在原文裡，連學習率等價於參數化這件事，Jacot 等人自己都註記了。

即便如此，社群還是需要另外一篇 NeurIPS 論文來把「這是縮放造成的」說清楚，並給它一個名字；之後又花了好幾年，才把「這個定理到底說了什麼、沒說什麼」消化乾淨。

糾正的成本遠高於犯錯的成本。而在文獻裡，一個正確的證明配上一個錯誤的故事，可以跑得比任何人預期的都遠。

這背後有一個結構性的誘因：可證明性本身就是篩選標準。一個把想理解的現象關掉、因而變得可以證明的設定，會拿到不成比例的注意力；而要糾正它，還得再寫一篇同樣可以被證明的論文。

寫這篇的同一個月，數學界正在對 AI 提出形狀相同的抱怨。9 月 11 日，26 位 Fields 獎得主聯名發表〈[A Severe Misalignment of AI in Mathematics](https://mathandai.org)〉，核心是一句話：

> Solving problems is only a tool and proxy for achieving the primary goal of conceptual understanding and insight.

他們擔心的是解答被急著公布，來不及好好寫下來、提煉方法、引用前人，於是「數學家之間那條關鍵的人類傳遞鏈」會斷掉。

值得注意的是，NTK 這件事裡沒有 AI。從頭到尾都是人：人證明了定理，人把它轉述成另一個故事，人花了好幾年才消化回來。所以那條傳遞鏈要防的不只是 AI——只要誘因是「能被證明、能被檢驗」，社群自己就會走上同一條路。

而把事情修回來的，也正是那條鏈：有人回去讀原文，有人替現象重新命名，有人做實驗證明那個故事站不住。NTK 至少還有人回頭拆。等到每週都冒出一個新證明的時候，還會有人回頭嗎？

## 參考資料

- Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572v4). NeurIPS 2018.
- Chizat, L., Oyallon, E., & Bach, F. (2019). [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956v5). NeurIPS 2019.
- Du, S. S., Zhai, X., Póczos, B., & Singh, A. (2019). [Gradient Descent Provably Optimizes Over-parameterized Neural Networks](https://arxiv.org/abs/1810.02054v2). ICLR 2019.
- Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., & Wang, R. (2019). [On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/abs/1904.11955v2). NeurIPS 2019.
- Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2021). [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065v3). ICLR 2021.
- Rahimi, A., & Recht, B. (2007). [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf). NIPS 2007.
- Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz Mappings into a Hilbert Space. *Contemporary Mathematics*, 26, 189–206.
- Fields 獎得主聯名 (2026). [A Severe Misalignment of AI in Mathematics](https://mathandai.org). 2026 年 9 月 11 日。
