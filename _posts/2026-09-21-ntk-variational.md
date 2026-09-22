---
title: "兩個 δ"
subtitle: "用一階變分重走一遍 NTK"
date: 2026-09-21
tags: [ntk, kernel-methods, calculus-of-variations, math]
description: "〈Deep Learning 的無窮維煙花〉批評 NTK 的結論是設定造成的，這篇沿著原文的推理，找出是哪一步走岔了。把網路寫成積分，只取一階變分：「權重不動」是配比混進位移的結果，kernel 裡的 feature 項一直都在。再拿這張地圖回看 2019 年之後的後續研究：吵了七年的 lazy 和 rich，是同一隻大象的不同部位。"
---

## 一、前言

在[〈Deep Learning 的無窮維煙花〉](/posts/infinite-width-fireworks/)裡，我們批評 NTK 的結論是設定造成的。但這個結論是怎麼一路推出來的？這篇想沿著原文的推理走一遍，找出是哪一步走岔了，讓設定被說成了網路的特性。

原文的符號繁瑣，推導冗長。不過用我們在小學三年級學到的微積分，就能把平行堆疊的無窮維函數寫成一個簡單的積分；再用我們在小學四年級學到的分析技巧，不必真的把積分算出來，就能拿掉積分符號，直接讀出結論。

## 二、兩個 δ

這一章只用一階變分。網路寫成積分之後，$$f$$ 的變動只有兩個來源：$$\delta w$$，feature mapping 在動；$$\delta\mu$$，配比在動。接下來分四步走：先把網路和損失函數寫成積分，再看最佳化條件裡 $$w$$ 還在不在，然後拿 NTK 的「權重不動」來對照，最後接回 kernel。一路上要盯著的，是這兩個 $$\delta$$ 各自怎麼被選，以及配比放在哪裡。

### Step 1　函數平行堆疊的積分形式

網路是一堆函數並排加起來：

$$
f(x;\theta)=\frac{1}{\sqrt n}\sum_{i=1}^{n} a_i\,\sigma(x;w_i).
$$

把歸一化係數和 $$a_i$$ 合起來，看成分配給 $$w_i$$ 的配比：

$$
\frac{a_i}{\sqrt n}\;\longrightarrow\;d\mu(w_i),
$$

就能改寫成積分：

$$
f(x)=\int_w \sigma(x;w)\,d\mu(w).
$$

這裡的 $$\mu$$ 就是我們熟悉的測度。注意到這個一般形式不在乎歸一化係數具體怎麼設：可以是 $$1/\sqrt n$$，可以是 $$1/n$$，也可以是任何說得通的分佈，只要能加總，就能這樣寫。

對於損失函數，我們也可以不失一般性地依樣畫葫蘆：

$$
L=\int_x \ell(f)\,d\rho(x),
$$

其中 $$\rho$$ 代表資料 $$x$$ 的密度，或者你要叫它測度。

### Step 2　一階變分下的最佳化條件：能不能拿掉 w

有了這兩條表達，我們試著寫下 $$L$$ 這個泛函在 $$f$$ 附近的變動：

$$
L[f+\delta f]-L[f]=\int_x \frac{\partial\ell}{\partial f}\,\delta f\,d\rho(x)+O(\delta f^2).
$$

這招就是我們在小學三年級物理課上學到的變分。

$$f$$ 同時依賴 $$\sigma$$ 裡的 $$w$$ 和配比 $$\mu$$，乘積法則給出兩項：

$$
\delta f(x)=\int_w \frac{\partial\sigma}{\partial w}(x;w)\cdot\delta w\,d\mu(w)+\int_w \sigma(x;w)\,d\delta\mu(w).
$$

第一項對應的是 feature mapping 的變動，第二項對應的是配比的變動。

把 $$\delta f$$ 代回去，交換積分順序，先對 $$x$$ 積：

$$
\delta L=\int_w \underbrace{\left[\int_x \frac{\partial\ell}{\partial f}\,\frac{\partial\sigma}{\partial w}\,d\rho\right]}_{g_w(w)}\cdot\,\delta w\,d\mu+\int_w \underbrace{\left[\int_x \frac{\partial\ell}{\partial f}\,\sigma\,d\rho\right]}_{g_\mu(w)}\,d\delta\mu .
$$

到目前為止，$$g_w$$ 還躺在式子裡。它跟寬度無關：不管 $$\mu$$ 是幾個點加起來、配比怎麼設，$$g_w$$ 都長這樣。這跟 NTK 說的對不上。

仔細看 $$g_w$$：殘差 $$\frac{\partial\ell}{\partial f}$$ 乘上 $$\frac{\partial\sigma}{\partial w}$$，對資料積分。$$\frac{\partial\sigma}{\partial w}$$ 是 $$\sigma$$ 給的，只要 $$\sigma$$ 依賴 $$w$$，$$g_w$$ 一般而言就不為零。這代表 feature mapping 變動的貢獻就在那裡。

$$g_w$$ 是問題給的，乘在它旁邊的 $$\delta w$$ 則是可以任意設定的。

假設 $$f$$ 是一個已經 train 好的類神經網路。仿造我們在小學三年級體育課學到的最小作用量原理：$$f$$ 是最佳，則一階變分為零：

$$
\delta L=\int_w g_w\cdot\delta w\,d\mu+\int_w g_\mu\,d\delta\mu=0 .
$$

要讓第一項為零，有兩條路：

- $$\delta w$$ 任取。那麼只能 $$g_w=0$$，這時才能拿掉積分符號。
- 設定 $$\delta w=0$$。這一項直接消失，$$g_w$$ 不受任何約束。

這代表 $$\delta w$$ 任選、feature mapping 的變動卻不提供貢獻，只有在一階變分為零的情況下才會出現。

第二項同理。兩項都任取，得到 $$g_w=0,\ g_\mu=0$$；設定 $$\delta w=0$$，只剩 $$g_\mu=0$$，也就是固定 feature mapping、只調配比的最佳條件。

### Step 3　比對 NTK 的結論與變分方法下的對應

NTK 的結論是：網路夠寬，權重幾乎不動，訓練等同固定 feature mapping。照 Step 2，要讓 feature mapping 的變動不提供貢獻，要嘛走到一階變分為零，要嘛設定 $$\delta w=0$$。原文兩者都沒做，那「不動」是從哪來的？

回到 Step 2 裡 $$w$$ 的那一項：

$$
\delta L_w=\int_w g_w\cdot\delta w\,d\mu .
$$

原文用的是求和的寫法。把 $$\mu$$ 的原子展開，同一個 $$\delta L_w$$ 就寫成

$$
\delta L_w=\sum_{i=1}^{n} m_i\,g_w(w_i)\cdot\delta w_i ,\qquad m_i=\frac{a_i}{\sqrt n}.
$$

兩種寫法完全相等，差別在「梯度」指的是 $$\delta w$$ 前面的哪個係數：

- 積分寫法：係數是 $$g_w$$。配比留在 $$d\mu$$ 裡，梯度就是對 $$\rho$$ 的積分，跟寬度無關。
- 求和寫法：係數是 $$m_i\,g_w(w_i)$$，也就是 $$\frac{\partial L}{\partial w_i}$$。配比從 $$d\mu$$ 被拿出來，併進了係數。

原文用梯度下降，位移取成步長乘上求和寫法的係數：

$$
\delta w_i=-\eta\,\frac{a_i}{\sqrt n}\,g_w(w_i).
$$

$$\eta$$ 固定，$$n$$ 越大，$$\delta w_i$$ 越小。

配比就這樣跟著係數，從 $$d\mu$$ 跑進了 $$\delta w$$。$$g_w$$ 從頭到尾沒有變；$$\delta w$$ 本來可以任意設定，這裡被這個選擇綁上了 $$1/\sqrt n$$。權重不動，是這個選擇的結果，不是 $$g_w$$ 消失了。

Step 1 說過，歸一化係數怎麼設都行。換成 $$1/n$$，梯度帶上 $$1/n$$；步長跟著放大 $$n$$ 倍，$$\delta w_i=-\eta\,a_i\,g_w(w_i)$$，跟寬度無關，權重就動起來了。同一個網路、同一個 $$g_w$$，feature mapping 變動的貢獻就看配比怎麼設。

到目前為止，我們還找不到理由排除 feature mapping 變動的貢獻。單從梯度下降的角度看，「權重不動」幾乎和我們在前文的批評一樣，已經被判了死刑。

### Step 4　連接回 kernel

接著我們更進一步，連接回 kernel，去找 NTK 的重頭戲：tangent kernel。

回到 Step 2 的 $$\delta L$$。$$\delta w$$ 和 $$\delta\mu$$ 可以任意設定，那就挑最省事的：照我們在小學五年級學到的梯度下降，沿著 $$g_w$$、$$g_\mu$$ 的反方向走，

$$
\delta w=-\eta\,g_w,\qquad d\delta\mu=-\eta\,g_\mu\,d\mu .
$$

代回 $$\delta f$$，把 $$g_w$$、$$g_\mu$$ 展開：

$$
\delta f(x)=-\eta\int_{x'}K(x,x')\,\frac{\partial\ell}{\partial f}(x')\,d\rho(x'),
$$

$$
K(x,x')=\underbrace{\int_w\frac{\partial\sigma}{\partial w}(x;w)\cdot\frac{\partial\sigma}{\partial w}(x';w)\,d\mu}_{K_{\delta w}}+\underbrace{\int_w\sigma(x;w)\,\sigma(x';w)\,d\mu}_{K_{\delta\mu}}.
$$

kernel 就這樣冒出來了，兩項都在：$$K_{\delta w}$$ 來自 feature mapping 的變動，$$K_{\delta\mu}$$ 來自配比的變動。

現在回到 NTK 的說法：權重不動，feature 側沒有貢獻。在 kernel 上，這句話就是 $$K_{\delta w}$$ 消失。

對照 Jacot 等人的 Theorem 1，極限核的遞迴是

$$
\Theta^{(L+1)}=\Theta^{(L)}\,\dot\Sigma^{(L+1)}+\Sigma^{(L+1)} .
$$

兩層時，$$\Sigma^{(2)}$$ 對應 $$K_{\delta\mu}$$，$$\Theta^{(1)}\dot\Sigma^{(2)}$$ 對應 $$K_{\delta w}$$。$$K_{\delta w}$$ 還在。

權重不動，feature 那一項卻留在 kernel 裡。這兩件事要怎麼同時成立？手上已經有兩條線索。

第一條來自 Step 2。kernel 要 non-trivial，代表 $$\frac{\partial\sigma}{\partial w}$$ 不能消失。它是 $$\sigma$$ 給的，不帶配比，寬度再大也不會讓它變小；對資料積分得到的 $$g_w$$ 也一樣。

第二條來自 Step 3。原文的位移帶著配比，$$\delta w_i=-\eta\,m_i\,g_w(w_i)$$。這個選擇就寫在原文定義 NTK 的那一行：

$$
\Theta^{(L)}(\theta)=\sum_{p=1}^{P}\partial_{\theta_p}F^{(L)}(\theta)\otimes\partial_{\theta_p}F^{(L)}(\theta),
$$

其中 $$\partial_{\theta_p}F$$ 是網路輸出對第 $$p$$ 個參數的偏導，$$\otimes$$ 是在兩個資料點上相乘。前一句是「For ANNs trained using gradient descent」：位移取成每個參數自己的偏導數，每個參數等權重相加。

兩條線索合起來，答案就出來了。$$\frac{\partial\sigma}{\partial w}$$ 和 $$g_w$$ 都不帶配比，不會消失；位移之所以趨於零，是因為配比被混進了位移。每個 $$\delta w_i$$ 帶著 $$1/\sqrt n$$，這是「權重不動」的來源。

所以「權重不動」不是網路的性質，是把配比混進位移的結果。把配比放回 $$d\mu$$，只要總質量不發散，每個 $$w$$ 的位移就是 $$-\eta\,g_w$$，和寬度無關。$$K_{\delta w}$$ 一直留在原文的 kernel 裡，feature 側從來沒有停止貢獻。

## 三、同一隻大象

第二章走完，NTK 的現象在變分下很平凡：配比混進了位移，個別權重看起來不動；feature 那一項一直留在 kernel 裡。手上有了這張地圖，這一章拿它去看 2019 年之後的後續研究。這些研究各自從一個角度切進來，指認一個旋鈕，也各自摸到了一塊。先看它們說了什麼、吵了什麼，最後再把整隻拿出來。

**凍結的 kernel 夠不夠用。**第一批工作把凍結當成工具，旋鈕是「讓權重留在初始值附近」。Du 等人（[2019](https://arxiv.org/abs/1810.02054v2)）證明過參數化網路的梯度下降會收斂到全域最小，關鍵是「over-parameterization and random initialization jointly restrict every weight vector to be close to its initialization」；Allen-Zhu 等人（[2019](https://arxiv.org/abs/1811.03962v5)）在初始值附近證明了網路和 NTK 等價。Arora 等人（[2019](https://arxiv.org/abs/1904.11955v2)）直接算出卷積網路的無窮寬 kernel，在 CIFAR-10 上拿到 77.43%，但離對應的 CNN 還差 5% 到 6%，他們自己說 NTK regime「cannot fully explain the success of neural networks yet」。Fort 等人（[2020](https://arxiv.org/abs/2010.15110v1)）量了訓練中的 NTK：前兩三個 epoch 它變得很快，「learning useful features from the training data」。Lee 等人（[2020](https://arxiv.org/abs/2007.15801v2)）發現 weight decay 和大學習率會「break the correspondence between finite and infinite networks」。

**凍結從哪裡來。**這一批工作各自指認一個旋鈕。Lee 等人（[2019](https://arxiv.org/abs/1902.06720v4)）指認寬度，「rather than the particular parameterization」，但附錄寫明兩種參數化要配上逐層調整的學習率才等價。Chizat、Oyallon 與 Bach（[2019](https://arxiv.org/abs/1812.07956v5)）指認輸出的縮放：凍結「is due to a choice of scaling, often implicit」，他們把它命名為 lazy regime，實驗顯示這個 regime 下的 CNN 表現變差。Yang 與 Hu（[2021](https://arxiv.org/abs/2011.14522v3)）指認參數化，也就是初始化方差、前乘係數、學習率各自隨寬度的冪次：標準參數化和 NTK 參數化「do not admit infinite-width limits that can learn features」，換成他們的 μP 才學得到；並且把二分寫成定理，每一種參數化「either admits feature learning or is in kernel regime, but not both」。一年後，μP 成了調大模型超參數的工具（[Yang 等人 2022](https://arxiv.org/abs/2203.03466v2)），理由之一是它讓每個參數「not stuck at initialization」。Yang 與 Littwin（[2023](https://arxiv.org/abs/2308.01814v2)）指認優化器：換成 Adam，二分仍然成立，只是 kernel 換成非線性的算子。

**兩個 regime 之間。**這一批工作轉一個旋鈕，看網路從一端走到另一端。Geiger 等人（[2019](https://arxiv.org/abs/1906.08034v4)）轉的是輸出的縮放 $$\alpha$$，發現分界在 $$\alpha^{\ast}\sim1/\sqrt h$$，$$h$$ 是寬度。Woodworth 等人（[2020](https://arxiv.org/abs/2002.09277v3)）轉的是初始化的大小，它「controls the transition between the "kernel" (aka lazy) and "rich" (aka active) regimes」。Lewkowycz 等人（[2020](https://arxiv.org/abs/2003.02218v1)）轉的是學習率：學習率夠大，kernel 就會動，「even at large width」。Kunin 等人（[2024](https://arxiv.org/abs/2406.06158v2)）轉的是層與層之間的相對尺度。

**kernel 怎麼動。**另一批工作接受 kernel 會動，去描述它怎麼動：自洽的場論（[Bordelon 與 Pehlevan 2022](https://arxiv.org/abs/2205.09653v3)）、一步梯度就長出的 spike（[Ba 等人 2022](https://arxiv.org/abs/2205.01445v1)）、average gradient outer product（[Radhakrishnan 等人 2024](https://arxiv.org/abs/2212.13881v3)）。Atanasov 等人（[2021](https://arxiv.org/abs/2111.00034v2)）發現，rich regime 訓練出來的網路，最後等價於「a kernel regression solution with the final network's tangent kernel」。

到了 2025 年，二分還在。有人說它太簡化，「this simple lazy–rich dichotomy overlooks a diverse underlying taxonomy of feature learning」（[Chou 等人 2025](https://arxiv.org/abs/2503.18114v2)）；有人論證在持續學習裡，寬度只有在「reduces the amount of feature learning, yielding more laziness」時才有好處（[Graldi 等人 2025](https://arxiv.org/abs/2506.16884v1)）。

七年下來，每一篇都摸到了一塊：寬度、縮放、參數化、初始化、層間比例、學習率、優化器。「kernel 和 feature learning 是兩個 regime」這個框架，一直沒有人拆。Yang 與 Hu 在正文裡還留下一句：「It may seem somewhat puzzling how the NTK limit induces change in f without feature or feature kernel evolution」。

現在把整隻大象拿出來。它就是 Step 2 的那條式子，配上 Step 4 的 kernel：

$$
\delta f(x)=\int_w \frac{\partial\sigma}{\partial w}(x;w)\cdot\underbrace{\delta w}_{\text{②}}\,\underbrace{d\mu(w)}_{\text{①}}+\int_w \sigma(x;w)\,d\delta\mu(w),
\qquad
K_{\delta w}(x,x')=\int_w \underbrace{\frac{\partial\sigma}{\partial w}(x;w)\cdot\frac{\partial\sigma}{\partial w}(x';w)}_{\text{③}}\,d\mu(w).
$$

① 是配比的尺度，② 是位移怎麼選，③ 是 $$\frac{\partial\sigma}{\partial w}$$ 在哪裡取值，初始值還是當下。每一個旋鈕都落在這三塊之一：

| 研究 | 旋鈕 | 落在哪一塊 |
|---|---|---|
| Du；Allen-Zhu；Arora（CNTK）；Lee 2019 的線性化 | 權重留在初始值附近 | ③ 停在初始值 |
| Lee 2019 | 寬度、逐層學習率 | ① 和 ② 的相對大小 |
| Chizat；Geiger | 輸出縮放 $$\alpha$$ | ①，分界 $$\alpha^{\ast}\sim1/\sqrt h$$ 正是 $$1/\sqrt n$$ 的配比 |
| Yang–Hu；μTransfer | 參數化的冪次 | ① 和 ② 隨寬度的冪次 |
| Yang–Littwin | 優化器 | ② |
| Woodworth；Kunin | 初始化大小、層間比例 | ① 的初始值 |
| Lewkowycz；Lee 2020 | 學習率 | ② |
| Fort；Atanasov；Bordelon；Ba；Radhakrishnan | kernel 怎麼動 | ③ 在當下取值時的軌跡 |

結論只有一句：只要 kernel 裡有 $$K_{\delta w}$$，feature learning 就有貢獻。所有的旋鈕，調的都是 $$K_{\delta w}$$ 的貢獻有多大，或者它停在哪裡；沒有一個能在訓練進行時把它拿掉，除非直接設定 $$\delta w=0$$。Yang 與 Hu 的困惑，來自配比混進了位移：這樣一來，只看得到每個權重幾乎不動。把配比放回 $$d\mu$$，$$f$$ 的變化就是 $$K_{\delta w}$$ 作用在殘差上，沒有什麼好困惑的。吵了七年的兩個 regime，是這三塊的不同設定，不是網路的兩種性質。

## 後記

前言問，作者們在這條路上看到了什麼。能回答的只有原文寫下的東西：他們的 kernel 裡寫著 $$K_{\delta w}$$，feature 一直有貢獻。「權重不動」是配比混進位移的結果，原文自己在 Remark 1 也說，$$1/\sqrt n$$ 的「side-effect」是讓權重的影響大幅變小，實驗時再用學習率 1.0 去補回來。

這件事其實更早就有人寫過。1995 年，Neal 在博士論文裡用同一個 $$1/\sqrt n$$ 推出無窮寬網路的 Gaussian process 極限，接著就說這個極限令人失望：「with Gaussian priors the contributions of individual hidden units are all negligible, and consequently, these units do not represent "hidden features"」。他的做法是換掉先驗，讓一部分單元在無窮寬下仍然保有不可忽略的權重。Jacot 等人為 GP 那一步引了 Neal 的書。

回頭看第三章的七年，與其說大家在瞎忙，不如說工具選錯了。在參數空間裡對每個參數求梯度，配比天生會混進位移；在這個視角下，每轉一個旋鈕都像發現一個新的 regime，也就需要一個新名字、一條新的分界。把網路寫成積分，配比留在測度裡，同樣的現象只剩下兩個 $$\delta$$ 怎麼選。工具選對了，很多問題不是被解決，而是根本不會出現。

最後想起那個老故事。幾個人在黑暗裡摸一頭大象：摸到鼻子的說它像蛇，摸到腿的說它像柱子，摸到耳朵的說它像扇子，摸到尾巴的說它像繩子。每個人都沒有說錯，他們手上摸到的確實是那個樣子；錯的是以為自己摸到了整頭象，然後為「大象到底像蛇還是像柱子」吵了起來。摸到初始值那一塊的說 kernel 是凍結的，摸到配比的說是縮放，摸到步長的說是學習率，摸到參數化的說是冪次。他們摸到的都是真的。

需要的不是更靈巧的手，而是把燈打開。

## 參考資料

- Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572v4). NeurIPS 2018.
- Neal, R. M. (1995). [Bayesian Learning for Neural Networks](https://www.cs.toronto.edu/~radford/ftp/thesis.pdf). PhD thesis, University of Toronto.
- Du, S. S., Zhai, X., Póczos, B., & Singh, A. (2019). [Gradient Descent Provably Optimizes Over-parameterized Neural Networks](https://arxiv.org/abs/1810.02054v2). ICLR 2019.
- Allen-Zhu, Z., Li, Y., & Song, Z. (2019). [A Convergence Theory for Deep Learning via Over-Parameterization](https://arxiv.org/abs/1811.03962v5). ICML 2019.
- Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., & Wang, R. (2019). [On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/abs/1904.11955v2). NeurIPS 2019.
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720v4). NeurIPS 2019.
- Chizat, L., Oyallon, E., & Bach, F. (2019). [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956v5). NeurIPS 2019.
- Geiger, M., Spigler, S., Jacot, A., & Wyart, M. (2019). [Disentangling feature and lazy training in deep neural networks](https://arxiv.org/abs/1906.08034v4).
- Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D., & Srebro, N. (2020). [Kernel and Rich Regimes in Overparametrized Models](https://arxiv.org/abs/2002.09277v3). COLT 2020.
- Lewkowycz, A., Bahri, Y., Dyer, E., Sohl-Dickstein, J., & Gur-Ari, G. (2020). [The large learning rate phase of deep learning: the catapult mechanism](https://arxiv.org/abs/2003.02218v1).
- Fort, S., Dziugaite, G. K., Paul, M., Kharaghani, S., Roy, D. M., & Ganguli, S. (2020). [Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel](https://arxiv.org/abs/2010.15110v1). NeurIPS 2020.
- Lee, J., Schoenholz, S. S., Pennington, J., Adlam, B., Xiao, L., Novak, R., & Sohl-Dickstein, J. (2020). [Finite Versus Infinite Neural Networks: an Empirical Study](https://arxiv.org/abs/2007.15801v2).
- Yang, G., & Hu, E. J. (2021). [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522v3). ICML 2021.
- Atanasov, A., Bordelon, B., & Pehlevan, C. (2021). [Neural Networks as Kernel Learners: The Silent Alignment Effect](https://arxiv.org/abs/2111.00034v2).
- Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., & Gao, J. (2022). [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466v2).
- Bordelon, B., & Pehlevan, C. (2022). [Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks](https://arxiv.org/abs/2205.09653v3). NeurIPS 2022.
- Ba, J., Erdogdu, M. A., Suzuki, T., Wang, Z., Wu, D., & Yang, G. (2022). [High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation](https://arxiv.org/abs/2205.01445v1).
- Yang, G., & Littwin, E. (2023). [Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit](https://arxiv.org/abs/2308.01814v2).
- Kunin, D., Raventós, A., Dominé, C., Chen, F., Klindt, D., Saxe, A., & Ganguli, S. (2024). [Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning](https://arxiv.org/abs/2406.06158v2). NeurIPS 2024.
- Radhakrishnan, A., Beaglehole, D., Pandit, P., & Belkin, M. (2024). [Mechanism for feature learning in neural networks and backpropagation-free machine learning models](https://arxiv.org/abs/2212.13881v3). Science, 383(6690), 1461–1467.
- Chou, C.-N., Le, H., Wang, Y., & Chung, S. (2025). [Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry](https://arxiv.org/abs/2503.18114v2). ICML 2025.
- Graldi, J., Breccia, A., Lanzillotta, G., Hofmann, T., & Noci, L. (2025). [The Importance of Being Lazy: Scaling Limits of Continual Learning](https://arxiv.org/abs/2506.16884v1). ICML 2025.
