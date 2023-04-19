最大后验方法寻找的是后验概率分布 $p(X(\omega) \mid Y(\omega))$ 的最大值。很明显, 对 一个单峰且对称的分布而言 (如高斯分布)，最大后验和贝叶斯方法是等价的。然 在某些场景中，如果信号的分布难以做出假设，使用最大后验方法往往比使用贝口 斯方法要简单一些，因为一个末知分布的最大值通常要比其均值更容易获取。
在这一节中, 主要介绍应用最广泛的贝叶斯准则下的 MMSE 方法（Loizou， 2013)。贝叶斯准则下的 MMSE 最优解和维纳滤波的区别在于:
（1）维纳滤波法有线性假设, 即认为干净语音和带噪语音之间存在线性关系: $X(\omega)=Y(\omega) H(\omega)$ 。而贝叶斯降噪没有这一假设, 完全通过统计特性来求解干净语 音的最优估计 $\hat{X}(\omega)$, 但是这里需要对语音信号的概率分布做出另外的假设。
（2）维纳滤波法寻找的是复数谱意义上的最优解，即最优的 $X(\omega)$ 。下面我们 寻找幅度谱上的最优解, 即最优的 $|X(\omega)|$ 。
定义基于幅度谱的 MMSE 优化目标:
$$
e=E\left[\left(\hat{X}_k-X_k\right)^2\right]
$$
为了表述简洁, 这里令 $\hat{X}_k=\left|\hat{X}\left(\omega_k\right)\right|$, 即估计的干净语音在第 $k$ 个频点上的幅度。
MMSE 估计器的目标是使每个频点的幅度与真实幅度之间的平方误差的数学期望最 小。另外, 令 $\boldsymbol{Y}=\left[Y\left(\omega_1\right), Y\left(\omega_2\right), \cdots, Y\left(\omega_N\right)\right]$, 表示带噪信号在所有频点上的频谱, 那么在贝叶斯 MSE 准则下, 该数学期望需要通过 $X_k$ 与 $\boldsymbol{Y}$ 之间的联合概率密度函数 $p\left(X_k, \boldsymbol{Y}\right)$ 来求解, 即
$$
e=\iint\left(\hat{X}_k-X_k\right)^2 p\left(X_k, \boldsymbol{Y}\right) \mathrm{d} \boldsymbol{Y} \mathrm{d} X_k
$$
使上式最优的 $\hat{X}_k$ 为
$$
\hat{X}_k=\int x_k p\left(x_k \mid \boldsymbol{Y}\right) \mathrm{d} x_k=E\left[X_k \mid \boldsymbol{Y}\right]
$$


---


也就是说最优解就是 $X_k$ 在条件 $\boldsymbol{Y}$ 下的后验数学期望, 或者说是后验概率密度函 数 $p\left(x_k \mid \boldsymbol{Y}\right)$ 在全体 $x_k$ 上的均值。
为了求解这个问题, 需要做出两点假设: (1)假设语音信号的频谱（实部和虚部）
这两点假设事实上都是极大简化的, 因为实际的语音信号往往并不满足这两个 条件, 但是基于这两个假设得到的降噪方法在试验中被证明是有效的, 故这里先不 去过多探讨它们的合理性。基于第二点假设，问题变为
$$
\hat{X}_k=E\left[X_k \mid Y(\omega)\right]=\int x_k p\left(x_k \mid Y(\omega)\right) \mathrm{d} x_k=\frac{\int x_k p\left(Y(\omega) \mid x_k\right) p\left(x_k\right) \mathrm{d} x_k}{p\left(Y(\omega) \mid x_k\right) p\left(x_k\right) \mathrm{d} x_k}
$$
上面等式的最后一步是由贝叶斯条件概率定律推导得出的。求解该式的关键问题是 $p\left(Y(\omega) \mid x_k\right)$, 也就是在 $x_k$ 条件下 $Y(\omega)$ 的条件概率。根据前述第一点假设, $Y(\omega)$ 可 以被认为是 $X(\omega)$ 和 $N(\omega)$ 两个高斯分布随机变量的和, 那么这个条件概率 $p\left(Y(\omega) \mid x_k\right)$ 依然是满足高斯分布的, 并且其均值是 $X(\omega)$, 而方差是 $N(\omega)$ 的方差,
$$
p\left(Y(\omega) \mid x_k\right)=\frac{1}{\pi \lambda_d(k)} \exp \left(-\frac{1}{\lambda_d(k)}|Y(\omega)-X(\omega)|^2\right)
$$
其中, $\lambda_d(k)=E\left[\left|N\left(\omega_k\right)\right|^2\right]$ 为第 $k$ 个频点上的噪声功率谱的期望。此外, $p\left(x_k\right)$ 也满 足高斯分布, 即
在
$$
p\left(x_k\right)=\frac{1}{\pi \lambda_x(k)} \exp \left(-\frac{x_k^2}{\lambda_x(k)}\right)
$$
其中, $\lambda_x(k)=E\left[\left|X\left(\omega_k\right)\right|^2\right]$ 为第 $k$ 个频点上的语音功率谱的期望。将以上两个概 率分布代入前述积分公式, 进行化简后可得贝叶斯 MMSE 估计器的最终计算方法 (过程略):
$$
\hat{X}_k=\sqrt{\lambda(k)} \Gamma(1.5) \Phi\left(-0.5,1 ;-v_k\right)
$$
其中， $\Gamma$ 和 $\Phi$ 分别表示伽马 (Gamma) 函数和合流超几何 (Confluent Hypergeometric) 函数, 其定义分别为
也就是说最优解就是 $X_k$ 在条件 $\boldsymbol{Y}$ 下的后验数学期望, 或者说是后验概率密度函 数 $p\left(x_k \mid \boldsymbol{Y}\right)$ 在全体 $x_k$ 上的均值。
为了求解这个问题, 需要做出两点假设: (1)假设语音信号的频谱（实部和虚部）
这两点假设事实上都是极大简化的, 因为实际的语音信号往往并不满足这两个 条件, 但是基于这两个假设得到的降噪方法在试验中被证明是有效的, 故这里先不 去过多探讨它们的合理性。基于第二点假设，问题变为
$$
\hat{X}_k=E\left[X_k \mid Y(\omega)\right]=\int x_k p\left(x_k \mid Y(\omega)\right) \mathrm{d} x_k=\frac{\int x_k p\left(Y(\omega) \mid x_k\right) p\left(x_k\right) \mathrm{d} x_k}{p\left(Y(\omega) \mid x_k\right) p\left(x_k\right) \mathrm{d} x_k}
$$
上面等式的最后一步是由贝叶斯条件概率定律推导得出的。求解该式的关键问题是 $p\left(Y(\omega) \mid x_k\right)$, 也就是在 $x_k$ 条件下 $Y(\omega)$ 的条件概率。根据前述第一点假设, $Y(\omega)$ 可 以被认为是 $X(\omega)$ 和 $N(\omega)$ 两个高斯分布随机变量的和, 那么这个条件概率 $p\left(Y(\omega) \mid x_k\right)$ 依然是满足高斯分布的, 并且其均值是 $X(\omega)$, 而方差是 $N(\omega)$ 的方差,
$$
p\left(Y(\omega) \mid x_k\right)=\frac{1}{\pi \lambda_d(k)} \exp \left(-\frac{1}{\lambda_d(k)}|Y(\omega)-X(\omega)|^2\right)
$$
其中, $\lambda_d(k)=E\left[\left|N\left(\omega_k\right)\right|^2\right]$ 为第 $k$ 个频点上的噪声功率谱的期望。此外, $p\left(x_k\right)$ 也满 足高斯分布, 即
在
$$
p\left(x_k\right)=\frac{1}{\pi \lambda_x(k)} \exp \left(-\frac{x_k^2}{\lambda_x(k)}\right)
$$
其中, $\lambda_x(k)=E\left[\left|X\left(\omega_k\right)\right|^2\right]$ 为第 $k$ 个频点上的语音功率谱的期望。将以上两个概 率分布代入前述积分公式, 进行化简后可得贝叶斯 MMSE 估计器的最终计算方法 (过程略):
$$
\hat{X}_k=\sqrt{\lambda(k)} \Gamma(1.5) \Phi\left(-0.5,1 ;-v_k\right)
$$
其中， $\Gamma$ 和 $\Phi$ 分别表示伽马 (Gamma) 函数和合流超几何 (Confluent Hypergeometric) 函数, 其定义分别为

---

$$
\begin{gathered}
\Gamma(x)=\int_0^{\infty} t^{x-1} \mathrm{e}^{-t} \mathrm{~d} t \\
\Phi(a, b ; z)=1+\frac{a}{b} \frac{z}{1 !}+\frac{a(a+1)}{b(b+1)} \frac{z}{2 !}+\frac{a(a+1)(a+2)}{b(b+1)(b+2)} \frac{z}{3 !}+\cdots
\end{gathered}
$$
而 $\lambda_k$ 和 $v_k$ 分别为
$$
\begin{gathered}
\lambda_k=\frac{\lambda_x(k) \lambda_d(k)}{\lambda_x(k)+\lambda_d(k)}=\frac{\lambda_x(k)}{1+\xi_k} \\
v_k=\frac{\xi_k}{1+\xi_k} \gamma_k
\end{gathered}
$$
其中, $\xi_k=\xi\left(\omega_k\right)$ 为先验信噪比, $\gamma_k=\gamma\left(\omega_k\right)$ 为后验信噪比。在实际应㣙, 噪比较难直接获取, 通常使用当前帧的后验信噪比减去 1 , 然后再与上-1 信噪比估计进行平滑处理, 作为当前帧的先验信噪比估计。 形式:
$$
\hat{X}_k=H_{\mathrm{MMSE}} Y_k
$$
而
$$
H_{\text {MWSE }}=\frac{\sqrt{\pi}}{2} \frac{\sqrt{v_k}}{\gamma_k} \exp \left(-\frac{v_k}{2}\right)\left[\left(1+v_k\right)\right] I_0\left(\frac{v_k}{2}\right)+v_k I_1\left(\frac{v_k}{2}\right)
$$
就是贝叶斯 MMSE 估计器的增益函数, 其中 $I_0(x)$ 和 $I_1(x)$ 分别为零阶和尧) 寒尔函数 (Modified Bessel Function)。其定义由下式给出:
$$
I_v(x)=j^{-v} J_v(j x)=\sum_{m=0}^{\infty} \frac{x^{v+2 m}}{2^{v+2 m} m ! \Gamma(v+m+1)}
$$

---

其中, $j \hat{\theta}_{x k}$ 为第 $k$ 个频点所估计的相位。对上式使用拉格朗日乘子法进行求解后可 得（过程略）最优的 $j \hat{\theta}_{x k}$ 为
$$
j \hat{\theta}_{x k}=j \theta_{y k}
$$
其中, $j \theta_{y k}$ 为带噪语音在第 $k$ 个频点上的相位。也就是说, 在前述两个假设下, 最 佳相位的贝叶斯 MMSE 估计就是带噪语音的相位。这一结论也为在一般的语音降噪 算法中只处理幅度而直接使用带噪语音相位的做法提供了理论支撑。
使用 MMSE 作为标准的最优化方法, 虽然在数学上完全成立并且也比较容易处 理, 然而如果考虑到人耳的听觉特性, 因为人耳对音量的感知和音频信号的能量之 间并非线性, 而是接近对数的关系, 所以 MMSE 准则在主观听感上并不一定是最优 解。语音信号的动态范围相当宽, 高能量和低能量之间往往有数量级的差异。这种 差异会使得低能量段产生的误差对整体误差的贡献非常低, 几乎可以被忽略, 然而 低能量段中的这些很小的误差却可以被人耳感知到。针对这个问题, 需要提出一种 更符合人耳听觉特性的准则函数, 使数学计算和主观听感匹配起来。一个比较典型 和常用的准则函数是对数 MMSE (log-MMSE)。它通过将信号的幅度谱变换到对数 域, 压缩了其动态范围, 使得低能量段和高能量段对整体误差的贡献更为均衡。 log-MMSE 的误差函数定义如下:
$$
e=E\left\{\left[\log \left(\hat{X}_k\right)-\log \left(X_k\right)\right]^2\right\}
$$
我们可以从另一个角度来看这个定义。因为
$$
\log \left(\hat{X}_k\right)-\log \left(X_k\right)=\log \left(\frac{\hat{X}_k}{X_k}\right)
$$
所以优化误差函数事实上是在优化估计的语音幅度谱与真实的语音幅度谱之间的比 值, 并且其目标是达到 1 , 这样得到的误差函数为 0 。
采用与 MMSE 估计器同样的思路, log-MMSE 估计器的最优解为
$$
\hat{X}_k=\exp \left(E\left[\log \left(X_k\right) Y\right]\right)
$$
使用与 MMSE 估计器相同的假设与类似的求解思路, 可以得到以下闭式解 (过程略)
$$
E\left[\log \left(X_k\right) \mid \boldsymbol{Y}\right]=\frac{1}{2}\left(\log \lambda_k+\log v_k+\int_{v_k}^{\infty} \mathrm{e}^{-t} \mathrm{~d} t\right)=\frac{1}{2}\left(\log \lambda_k+\log v_k-E i\left(-v_k\right)\right)
$$

---

其中, $\lambda_k$ 和 $v_k$ 已在 MMSE 估计器的推导过程定义, 而 $E i(x)$ 为指数稆分 Integral)。
$$
E i(x)=-\int_{-x}^{\infty} \frac{\mathrm{e}^{-t}}{t} \mathrm{~d} t
$$
最后, 对 $E\left[\log \left(X_k\right) \mid \boldsymbol{Y}\right]$ 求指数可得
$$
\hat{X}_k=\exp \left(E\left[\log \left(X_k\right) \mid \boldsymbol{Y}\right]\right)=\frac{\xi_k}{\xi_k+1} \exp \frac{-E i\left(-v_k\right)}{2} Y_k=H_{\log -M M S E} Y_k
$$
其中
$$
H_{\log -\mathrm{MMSE}}=\frac{\xi_k}{\xi_k+1} \exp \frac{-E i\left(-v_k\right)}{2}
$$
这就是 log-MMSE 估计器的增益函数。
通过对两组增益函数进行比较, 可以得知, 在相同的先验信噪比 $\xi_k$ 䄷 比 $\gamma_k$ 的前提下, log-MMSE 估计器的衰减量更大。而实验也表明, 和 MMSE| 相比, log-MMSE 估计器能够在更好地保持语音质量的前提下更多地㧜牭 观听感也更好。此外, 可以看到 log-MMSE 估计器的增益函数比 MMSE䫝 因此, log-MMSE 估计器在实际场景中被更广泛地使用。