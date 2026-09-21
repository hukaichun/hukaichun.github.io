---
title: "兩個 δ"
subtitle: "用一階變分重走一遍 NTK"
date: 2026-09-21
tags: [ntk, kernel-methods, calculus-of-variations, math]
description: "〈Deep Learning 的無窮維煙花〉批評了 NTK 的設定，這篇想知道作者們在這條路上看到了什麼。不取極限，把寬網路寫成參數空間上的積分，只寫一階變分，一步一步看設定拿掉了什麼、留下了什麼。"
---

[〈Deep Learning 的無窮維煙花〉](/posts/infinite-width-fireworks/)批評 NTK 的設定：$$1/\sqrt n$$ 加上固定步長，讓個別權重的更新隨寬度消失，「權重不動」是設定出來的。我們認為這個批評成立，但還是想知道作者們在這條路上究竟看到了什麼。

原文透過逐層取極限逼近無窮寬，推導很長。要看設定做了什麼，也許不需要那套極限：有限寬網路本來就能寫成參數空間上的積分，只是測度是原子的；把測度放寬，就是無窮維的版本。這篇從這個積分形式出發，只寫一階變分，走一步看一步。

## 1. 把網路寫成積分

單隱藏層網路

$$
f(x;\theta)=\frac{1}{\sqrt n}\sum_{i=1}^{n} a_i\,\sigma(x;w_i)
$$

對每個 $$n$$ 都可以寫成參數空間上的積分：

$$
f(x)=\int_w \sigma(x;w)\,d\mu(w),\qquad \mu=\frac{1}{\sqrt n}\sum_{i=1}^{n} a_i\,\delta_{w_i}.
$$

歸一化係數和讀出權重都吸收進 $$\mu$$。$$\mu$$ 是帶號測度，但做 Jordan 分解 $$\mu=\mu^+-\mu^-$$、把符號併進參數 $$(s,w)$$，$$s\in\{\pm1\}$$，就可以取 $$\mu\ge0$$。有限寬是原子測度的特例；$$\mu$$ 取一般的測度，就是無窮維的版本。

寫成這樣之後，眼前有兩個測度：參數上的 $$\mu$$，和資料上的 $$\rho$$。損失是

$$
L=\int_x \ell(f)\,d\rho.
$$

先不管動力學，也不管寬度怎麼趨近無限。只問一件事：這個 $$L$$ 動一下，會怎麼動？

## 2. 動一下

$$
\delta L=\int_x \frac{\partial\ell}{\partial f}\,\delta f\,d\rho .
$$

$$f$$ 同時依賴 $$\sigma(\cdot;w)$$ 裡的 $$w$$ 和測度 $$\mu$$，乘積法則給出兩項：

$$
\delta f(x)=\int_w \nabla_w\sigma(x;w)\cdot\delta w\;d\mu(w)\;+\;\int_w \sigma(x;w)\;d\delta\mu(w).
$$

兩項各有意思。第一項搬動質量：神經元在參數空間裡移動，也就是調特徵。第二項改變質量：每個神經元分到的權重增減，也就是調線性讀出。

把 $$\delta f$$ 代回 $$\delta L$$，得到兩個雙重積分，外層對資料 $$x$$、內層對參數 $$w$$：

$$
\delta L=\int_x \frac{\partial\ell}{\partial f}(x)\left[\int_w \nabla_w\sigma(x;w)\cdot\delta w\;d\mu(w)\right]d\rho(x)
\;+\;\int_x \frac{\partial\ell}{\partial f}(x)\left[\int_w \sigma(x;w)\;d\delta\mu(w)\right]d\rho(x).
$$

這樣寫，變分 $$\delta w$$、$$\delta\mu$$ 藏在內層。我們想看的是「往哪個方向動、$$L$$ 怎麼變」，所以把變分拉到外面：交換積分順序，先對 $$x$$ 積、再對 $$w$$ 積。$$\delta w(w)$$ 和 $$d\delta\mu(w)$$ 不依賴 $$x$$，可以提出內層：

$$
\delta L=\int_w \left[\int_x \frac{\partial\ell}{\partial f}(x)\,\nabla_w\sigma(x;w)\,d\rho(x)\right]\cdot\delta w\;d\mu(w)
\;+\;\int_w \left[\int_x \frac{\partial\ell}{\partial f}(x)\,\sigma(x;w)\,d\rho(x)\right]d\delta\mu(w).
$$

方括號裡只剩 $$w$$ 的函數，給它們名字：

$$
g_w(w)=\int_x \frac{\partial\ell}{\partial f}\,\nabla_w\sigma(x;w)\,d\rho,\qquad
g_\mu(w)=\int_x \frac{\partial\ell}{\partial f}\,\sigma(x;w)\,d\rho ,
$$

於是

$$
\delta L=\int_w g_w\cdot\delta w\;d\mu+\int_w g_\mu\;d\delta\mu .
$$

$$g_w$$、$$g_\mu$$ 都是殘差 $$\frac{\partial\ell}{\partial f}$$ 在資料測度 $$\rho$$ 下和特徵做內積：$$g_\mu(w)$$ 是殘差和第 $$w$$ 個特徵 $$\sigma(\cdot;w)$$ 的內積，$$g_w(w)$$ 是殘差和它的導數 $$\nabla_w\sigma(\cdot;w)$$ 的內積。資料只透過這兩個函數進到 $$\delta L$$ 裡。

有了 $$\delta L$$，可以先問最佳解長什麼樣子。不管用什麼方法走到那裡，最佳解都得滿足一個必要條件：往任何允許的方向動一下，$$L$$ 在一階都不變，也就是 $$\delta L=0$$ 對所有允許的變分成立。這是有限維「梯度為零」在這裡的版本。

乍看 $$\delta L=\int_x \frac{\partial\ell}{\partial f}\,\delta f\,d\rho=0$$，容易讀成逐點的條件：每個 $$x$$ 上，不是 $$\frac{\partial\ell}{\partial f}=0$$，就是 $$\delta f=0$$。但 $$\delta f$$ 不能任意指定，它只能是模型動得到的方向，所以條件只要求殘差在 $$L^2(\rho)$$ 裡正交於這些方向。模型動得到的方向越少，這個條件越弱。

哪些方向動得到，要看 $$(\delta w,\delta\mu)$$。兩者彼此獨立、各自可以任取，所以兩個係數要各自為零：

$$
g_w=0\quad(\mu\text{-a.e.}),\qquad g_\mu=0 .
$$

殘差正交於搬動特徵的方向，也正交於調整讀出的方向。這是個不太意外的結果。比較值得停下來看的是另一件事：**$$w$$ 一直都在。**

## 3. w 去哪了

NTK 流行的讀法是「寬網路的權重不動，所以等於固定特徵的核方法」。在上面的式子裡，要讓 $$w$$ 退場，得讓第一項消失。

第一項是導數乘位移：$$\nabla_w\sigma\cdot\delta w$$。導數是模型給的，$$\sigma(x;w)=\sigma(w^\top x)$$ 時就是 $$\sigma'(w^\top x)\,x$$，一般不為零，也跟寬度無關。位移是變分方向，由我們選。兩者在式子裡分得很清楚。

那原文是怎麼走到「權重不動」的？回到有限寬的參數寫法。原文用梯度下降，位移直接取成步長乘梯度：

$$
\delta w_i=-\eta\,\frac{\partial L}{\partial w_i}
=-\eta\,\frac{a_i}{\sqrt n}\int_x \frac{\partial\ell}{\partial f}\,\nabla_w\sigma(x;w_i)\,d\rho
=-\eta\,\frac{a_i}{\sqrt n}\,g_w(w_i).
$$

這裡的 $$a_i/\sqrt n$$ 原本是讀出權重，也就是 $$\mu$$ 在 $$w_i$$ 上的質量。位移一旦取成梯度，這個質量就從 $$\mu$$ 跑到了位移身上：$$\eta$$ 固定、$$n\to\infty$$，每個 $$\delta w_i$$ 都是 $$O(n^{-1/2})$$。推導走得很順，但走順的原因是位移和梯度被當成了同一個東西，梯度小，看起來就是網路不動。

在積分寫法裡，這三樣東西各在各的位置：$$\nabla_w\sigma$$ 是 $$O(1)$$，$$1/\sqrt n$$ 在 $$\mu$$ 裡，$$\delta w$$ 還沒被指定。第一項要消失，$$\sigma$$ 幫不上忙，只能設定 $$\delta w=0$$。

設定之後，駐點條件只剩 $$g_\mu=0$$，恰好是固定特徵 $$\sigma(\cdot;w)$$ 上線性讀出的最優條件。

這和〈無窮維煙花〉在步長上看到的是同一件事，換成了變分的語言：**$$\delta w$$ 是選的，$$\nabla_w\sigma$$ 才是 $$\sigma$$ 給的。**把位移取成梯度也是一種選法，它藏著什麼，下一節會看到。

不過到這裡還沒看到核。$$g_\mu=0$$ 是一個正交條件，不是一個核。核是從哪一步冒出來的？

## 4. 核在哪裡出現

$$\delta L$$ 對 $$(\delta w,\delta\mu)$$ 是線性泛函，要談最速下降，先得有內積。一般測度上沒有現成的 $$L^2$$ 內積，所以限制 $$\delta\mu=h\,\mu$$，只允許按比例增減已有的質量。這樣 $$\delta w$$ 與 $$h$$ 都是 $$w$$ 上的函數、都對 $$\mu$$ 積分，可以用同一個 $$L^2(\mu)$$：

$$
\delta L=\langle g_w,\delta w\rangle_{L^2(\mu)}+\langle g_\mu,h\rangle_{L^2(\mu)} ,
$$

最速下降方向是 $$\delta w=-\eta\,g_w$$、$$h=-\eta\,g_\mu$$。代回 $$\delta f$$，展開 $$g_w,g_\mu$$：

$$
\delta f(x)=-\eta\int_{x'}K(x,x')\,\frac{\partial\ell}{\partial f}(x')\,d\rho(x'),\qquad K=K_w+K_\mu,
$$

$$
K_w(x,x')=\int_w\nabla_w\sigma(x;w)\cdot\nabla_w\sigma(x';w)\,d\mu(w),\qquad
K_\mu(x,x')=\int_w\sigma(x;w)\,\sigma(x';w)\,d\mu(w).
$$

核出現了，是函數空間裡對 $$K$$ 做的梯度下降。兩項都在：$$K_w$$ 來自 $$\delta w$$，$$K_\mu$$ 來自 $$\delta\mu$$。

這個核和第 2 節交換積分順序是同一件事的兩面。把兩種特徵合寫成 $$\Phi(x,w)=(\sigma(x;w),\,\nabla_w\sigma(x;w))$$，同一個 $$\Phi$$ 有兩種積法：

$$
K(x,x')=\int_w \Phi(x,w)\cdot\Phi(x',w)\,d\mu(w),\qquad
G(w,w')=\int_x \Phi(x,w)\,\Phi(x,w')^{\top}\,d\rho(x).
$$

對 $$\mu$$ 積，得到資料上的核 $$K$$，它決定函數怎麼動；對 $$\rho$$ 積，得到參數上的算子 $$G$$，它決定參數怎麼動，包括 $$\delta w$$，也就是特徵怎麼動。有限維時這就是 $$JJ^\top$$ 與 $$J^\top J$$，非零譜相同。同一步更新，從參數那側看是在學特徵，從函數那側看是核梯度下降。

所以學特徵和核方法並不互斥：每一步都在學特徵，每一步也都是對當下的核 $$K_t$$ 做核梯度下降。特徵學習表現為核隨時間改變。要分清楚的是，每一步都是核梯度下降，不代表最後的解是某個核方法的解。整條軌跡用的是一串不同的核，得到的 $$f_T$$ 不是單一 RKHS 裡代表定理給出的那個解。

這裡的 $$\delta w=-\eta\,g_w$$ 可以和第 3 節對照。原文在參數空間用歐氏內積，得到 $$\delta w_i=-\eta\,(a_i/\sqrt n)\,g_w(w_i)$$，質量掛在位移上；這裡用 $$L^2(\mu)$$，質量被內積吸收，每個神經元的位移是 $$O(1)$$。「位移等於梯度」要先選定內積才有意義。

但質量沒有消失，它移到了 $$\delta f$$ 裡。$$K_w$$、$$K_\mu$$ 都對 $$\mu$$ 積分，而 $$1/\sqrt n$$ 縮放下 $$\mu$$ 的總質量是 $$\sum_i\lvert a_i\rvert/\sqrt n\sim\sqrt n$$。要讓 $$\delta f$$ 保持有限，$$\eta$$ 得跟著縮 $$1/\sqrt n$$，每個神經元的位移又回到 $$O(n^{-1/2})$$。換內積只是把這筆帳從位移搬到步長。真正決定權重動不動的，是 $$\mu$$ 的質量怎麼隨 $$n$$ 縮放，也就是參數化。

回頭數一下，為了走到這裡做了哪些選擇：

- **內積是選的。**給 $$\delta w$$ 與 $$\delta\mu$$ 各配一個權重，核就變成 $$c_1K_w+c_2K_\mu$$；換別的度量，形狀也跟著變。
- **$$\delta\mu=h\,\mu$$ 只能重新分配已有的質量。**$$\mu$$ 為零的地方生不出質量。
- **$$K$$ 依賴當下的 $$\mu$$ 與 $$w$$。**要得到整個訓練只用同一個核的核方法，還得把 $$K$$ 凍結在初始值。這一步推導本身沒有給。

變分條件本身不給核，核跟著選擇一起出現。這個模式在有限維裡應該也看得到，拿最熟悉的例子對一下。

## 5. 回頭看線性模型

$$f(x)=\phi(x)^\top\beta$$，資料 $$x_1,\dots,x_N$$，設計矩陣 $$\Phi\in\mathbb R^{N\times P}$$。平方損失的一階條件是

$$
\Phi^\top(\Phi\beta-Y)=0 ,
$$

它只說殘差正交於特徵，不推出 $$\beta\in\operatorname{span}\{\phi(x_j)\}$$。$$P>N$$ 時 $$\beta$$ 加上 $$\ker\Phi$$ 裡的任何向量都滿足同一個條件，卻會改變新點上的預測。核的結構來自**選哪一個解**：加 ridge、取最小範數，或從 $$0$$ 出發做梯度下降。選定之後 $$\beta=\Phi^\top c$$，預測只透過 $$k(x,x')=\phi(x)^\top\phi(x')$$ 依賴 $$\phi$$，這就是核方法。

果然是同一個模式：一階條件給正交，核來自額外的選擇。

把第 1 節的網路寫成這個形式：固定 $$w_i$$、只動讀出，特徵是 $$\phi(x)=(\sigma(x;w_i))_i$$，得到的核是 $$K_\mu$$。$$w_i$$ 也動的話，Jacobian 分成兩塊 $$J=[\,J_a\;J_w\,]$$，$$JJ^\top=J_aJ_a^\top+J_wJ_w^\top$$，對應 $$K_\mu+K_w$$。

走到這裡，手上有一個預期：如果「權重不動」是對的，NTK 應該只剩 $$K_\mu$$。可以翻開原文對一下了。

## 6. 原文的核長什麼樣

Jacot 等人（[2018](https://arxiv.org/abs/1806.07572v4)）的 Theorem 1 給出極限核的遞迴

$$
\Theta^{(L+1)}_\infty=\Theta^{(L)}_\infty\,\dot\Sigma^{(L+1)}+\Sigma^{(L+1)} .
$$

在兩層的情形，$$\Sigma^{(2)}$$ 對應讀出那一項 $$K_\mu$$，$$\Theta^{(1)}\dot\Sigma^{(2)}$$ 對應特徵那一項 $$K_w$$。兩邊的數值不完全相同：原文在參數上用歐氏內積，讀出那塊是 $$\frac1n\sum_i\sigma(x;w_i)\,\sigma(x';w_i)$$；我們在 $$L^2(\mu)$$ 下是 $$\int_w\sigma\,\sigma\,d\mu$$。權重的差別來自第 4 節選的內積，結構則相同：一項來自 $$\delta\mu$$，一項來自 $$\delta w$$。NTK 兩項都在。

這和預期不同。「權重不動」的讀法對應 $$\delta w=0$$，核是 $$K_\mu$$，也就是隨機特徵；原文的核卻保留了 $$\delta w$$ 的貢獻。在積分表示裡，$$\delta w=0$$ 和 $$K_w$$ 還在，兩者沒辦法同時成立。照原文的核，$$w$$ 在動。核在訓練中保持不變，靠的是第 3 節那一步：位移取成梯度，每個 $$w_i$$ 只動 $$O(n^{-1/2})$$，個別特徵的變化在極限裡消失。

Lee 等人（[2019](https://arxiv.org/abs/1902.06720v4)）在 Jacot 之後。他們沒有回頭檢查這一步，而是直接從「權重幾乎不動」出發，把網路換成它在初始參數的一階 Taylor 展開：

$$
f^{\rm lin}(x)=f_0(x)+\nabla_\theta f_0(x)\,(\theta-\theta_0).
$$

用積分寫，就是把第 2 節的 $$\delta f$$ 直接當成模型：

$$
f^{\rm lin}(x)=f_0(x)+\int_w\nabla_w\sigma(x;w)\cdot\delta w\;d\mu_0+\int_w\sigma(x;w)\;d\delta\mu ,
$$

$$\sigma$$ 與 $$\nabla_w\sigma$$ 停在初始值，$$\delta w$$、$$\delta\mu$$ 則當作有限大小的未知數。這是一個線性模型，特徵是 $$(\sigma,\nabla_w\sigma)$$。接下來由第 5 節，線性模型就是核方法，核是 $$K_\mu+K_w$$。這一段沒有用到任何關於神經網路的事。

跟神經網路有關的，只剩「為什麼可以換成一階展開」的論證（Lee 等人 Theorem 2.1）：學習率固定且小於臨界值，參數的總位移就有界；NTK 參數化又讓 Jacobian 的 Lipschitz 常數帶著 $$n^{-1/2}$$。兩者相乘，Jacobian 在訓練中幾乎不變。這兩個條件，一個是步長慣例，一個是參數化，正是第 3 節讓位移帶上 $$1/\sqrt n$$ 的那兩樣東西。

所以線性化並不是對「權重為什麼不動」的解釋。它把 Jacot 設定的後果當成前提，再用同一組設定證明這個前提成立。第 3 節看到位移和梯度被當成同一個東西；到了這裡，「權重不動」這個設定的後果，又被當成了寬網路本身的性質。

## 7. 帶著這張地圖看後續

還剩一個問題：核 $$K$$ 是在當下的 $$w$$ 上算的，$$w$$ 動了，核會不會跟著動？

這不需要動力學，對 $$K$$ 本身取一階變分就看得到。$$K$$ 透過 $$\Phi(\cdot;w)=(\sigma,\nabla_w\sigma)$$ 依賴 $$w$$：

$$
\delta_w K(x,x')=\int_w\Big[\big(\nabla_w\Phi(x;w)\,\delta w\big)\cdot\Phi(x';w)+\Phi(x;w)\cdot\big(\nabla_w\Phi(x';w)\,\delta w\big)\Big]\,d\mu(w).
$$

$$\delta\mu$$ 也會改變 $$K$$，但它只重新加權同一組特徵，張成的函數空間不變；特徵本身換掉，只來自 $$\delta w$$。所以這裡只看 $$\delta_wK$$。

$$\nabla_w\Phi$$ 是模型給的，一般不為零；$$\delta w\neq0$$，$$\delta_wK$$ 就一般不為零。核會不會動，跟第 3 節是同一個答案：看 $$\delta w$$ 怎麼選。至於在訓練中怎麼動、動多少，是動力學的問題，已經有不少實驗和分析在談，不是這篇要走的路。

把一路看到的整理成三種情形。三者都保留讀出的變分 $$\delta\mu=h\,\mu$$，差別在 $$\delta w$$，以及 $$\delta w$$ 帶來的核的變分 $$\delta_wK$$：

| | $$\delta w$$ | $$\delta_w K$$ | 核 | 對應 |
|---|---|---|---|---|
| (i) | $$=0$$ | $$=0$$ | $$K_\mu$$ | 隨機特徵；「權重不動」的讀法 |
| (ii) | $$\neq0$$ | 設為 $$0$$ | $$K_w+K_\mu$$，停在 $$w_0$$ | NTK；Lee 的線性化 |
| (iii) | $$\neq0$$ | $$\neq0$$ | $$K_w+K_\mu$$，隨 $$w_t$$ 改變 | feature learning |

寫成 $$\delta f$$：

- (i) 只剩讀出那一項：$$\delta f=\int_w\sigma(\cdot;w)\,d\delta\mu$$。$$\Phi$$ 沒動，$$\delta_wK=0$$ 自動成立。
- (ii) 兩項都在，但特徵停在初始值：$$\delta f=\int_w\nabla_w\sigma(\cdot;w_0)\cdot\delta w\,d\mu_0+\int_w\sigma(\cdot;w_0)\,d\delta\mu$$。$$\delta w\neq0$$，照上面的變分 $$\delta_wK$$ 一般不為零；這裡是把它設成零。
- (iii) 兩項都在，特徵在當下的 $$w_t$$ 取值：$$\delta f=\int_w\nabla_w\sigma(\cdot;w_t)\cdot\delta w\,d\mu_t+\int_w\sigma(\cdot;w_t)\,d\delta\mu$$，$$\delta_wK$$ 由上式給出。

(ii) 和 (iii) 的 $$\delta w$$ 一樣不為零，只差在 $$\delta_wK$$。$$\delta w\neq0$$ 而核不動，只能是設定。

拿這張表去看 NTK 之後的研究，有一批工作在做同一件事：指認是哪個設定讓 $$\delta_wK$$ 消失。

Chizat、Oyallon 與 Bach（[2019](https://arxiv.org/abs/1812.07956v5)）把網路輸出乘上一個大常數 $$\alpha$$。$$\alpha$$ 越大，權重只要動一點點，函數就能變很多；損失在權重還沒動到足以改變核之前就降下來了。他們把這叫 lazy training，摘要直說它「is due to a choice of scaling, often implicit」。這正是第 4 節最後看到的：權重動不動，由質量怎麼縮放決定。

Yang 與 Hu（[2021](https://arxiv.org/abs/2011.14522v3)）把這件事系統化。參數化是一整套規定：初始化的方差、乘在前面的係數、步長，各自是寬度的某個冪次。他們把其中穩定、而且不平凡的無窮寬極限做了分類：每一種極限要嘛特徵會動，要嘛是核梯度下降，「but not both」。這裡的核梯度下降指的是用固定的核。第 4 節的對偶顯示，每一步本來都是對當下的核做核梯度下降；所以二選一的對象不是核方法與特徵學習，而是核凍不凍結，「not both」照定義就成立。分類的內容在於哪些冪次落在哪一邊。這和第 4 節看到的一致：凍結 $$K$$ 不是推導給的，是參數化選出來的。他們也據此給出一套讓特徵在無窮寬下仍然會動的參數化。

Woodworth 等人（[2020](https://arxiv.org/abs/2002.09277v3)）換了一個旋鈕：初始化的大小。初始化大時，訓練停在核方法會選的那個解（第 5 節的最小範數解）；初始化小時，停在一個任何固定核都選不出來的解。第 5 節看到，核方法的解來自「選哪一個解」；這裡看到，這個選擇會隨設定改變。

三篇放在一起看：縮放、參數化、初始化大小，每一個旋鈕都能把 $$\delta_wK$$ 消掉，反過來調，也都能把它放回來。$$\delta_wK$$ 消不消失，是選出來的。

## 8. 回頭看一遍

把走過的路從頭排一次：

- **積分形式。**有限寬網路就是原子測度，放寬測度就是無窮維版本，沒有引進新的模型。
- **一階變分有兩項。**$$\delta w$$ 搬動特徵，$$\delta\mu$$ 調整讀出。$$\delta L=0$$ 是正交條件，不是逐點條件。
- **權重不動是選的。**$$\nabla_w\sigma$$ 是模型給的，$$\delta w$$ 是我們選的。原文把位移取成梯度，$$\mu$$ 的質量 $$1/\sqrt n$$ 就掛到了位移上。
- **核在選定內積之後才出現。**$$K$$ 與 $$G$$ 是同一個 $$\Phi$$ 的兩種積法，每一步都是對當下的核做核梯度下降；學特徵和核方法不互斥，互斥的是核凍不凍結。
- **線性模型的核來自選哪一個解。**一階條件只給正交。
- **原文的核兩項都在。**$$w$$ 在動；核不動，靠的是參數化和步長。線性化把這個後果當成了前提。
- **核會不會動，一階就看得出來。**後續研究指認的，是讓它不動的那些旋鈕。

回頭看，這些零件沒有一個是新的。原始與對偶是 SVM 教科書的標準推導：先寫原始問題，直接走到對偶，核只在對偶那側出現。Chapelle（[2007](https://people.csail.mit.edu/torralba/LabelMeToolbox/primalSVM/primal.pdf)）把兩側並排寫出來，$$X^\top X$$ 與 $$XX^\top$$ 給出同一個解，選哪一側只看哪個矩陣比較小。最小範數解、在當下參數做一階展開再解線性問題，也都是最佳化課本裡的東西。這篇做的只是把它們放回同一個積分式裡，看 NTK 的設定落在哪一格。

## 後記

開頭問作者們看到了什麼。能回答的只有原文寫下的東西：原文的核比流行的讀法多一項 $$K_w$$，$$w$$ 在動，而定理成立的條件讓核不動。定理沒有錯；它說的是那組設定，不是網路本身。

走這一趟，順手翻了後續的文獻，心裡有點彆扭。一般想像中，一個研究題目有意思，大概是這幾種：系統性地把 benchmark 往前推；給出新的理解，解釋過去解釋不了的東西；或者找到過去的理解說不通的現象。

NTK 之後有不少工作不在這三格裡。一種是做實驗確認核在訓練中會變，但核會不會變，一階變分就看得出來，看的是設定。這些實驗推翻的是流行的讀法，不是原文的數學；它們只對誤讀來說是新的。另一種是給已知的事取新名字：縮放決定特徵動不動，叫 lazy 和 rich；步長超過穩定上限時損失先衝高再掉下來，叫 [catapult](https://arxiv.org/abs/2003.02218v1)；「核凍不凍結」，被說成「核方法與特徵學習只能二選一」。

誤讀流行一天，確認它的實驗就有一天的市場。

## 參考資料

- Jacot, A., Gabriel, F., & Hongler, C. (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572v4). NeurIPS 2018.
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720v4). NeurIPS 2019.
- Chapelle, O. (2007). [Training a Support Vector Machine in the Primal](https://people.csail.mit.edu/torralba/LabelMeToolbox/primalSVM/primal.pdf). Neural Computation, 19(5), 1155–1178.
- Chizat, L., Oyallon, E., & Bach, F. (2019). [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956v5). NeurIPS 2019.
- Yang, G., & Hu, E. J. (2021). [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522v3). ICML 2021.
- Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D., & Srebro, N. (2020). [Kernel and Rich Regimes in Overparametrized Models](https://arxiv.org/abs/2002.09277v3). COLT 2020.
- Lewkowycz, A., Bahri, Y., Dyer, E., Sohl-Dickstein, J., & Gur-Ari, G. (2020). [The large learning rate phase of deep learning: the catapult mechanism](https://arxiv.org/abs/2003.02218v1).
