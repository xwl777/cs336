# Policy Gradient, TRPO, PPO, DPO, GRPO 原理与推导

这份笔记按一条主线展开：

```text
Policy Gradient
  -> TRPO: 限制策略更新不要太猛
  -> PPO: 用 clipped objective 近似 TRPO
  -> DPO: 不显式训练 reward model，直接从偏好数据优化 policy
  -> GRPO: 类 PPO，但去掉 value model，用 group relative reward 做优势估计
```

## 0. 基本设定

强化学习中有：

```text
状态: s
动作: a
策略: pi_theta(a | s)
奖励: r(s, a)
轨迹: tau = (s_0, a_0, s_1, a_1, ...)
折扣因子: gamma
```

目标是最大化期望累计奖励：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T} \gamma^t r(s_t, a_t)
\right]
$$

记一条轨迹的 return 为：

$$
R(\tau)
=
\sum_{t=0}^{T} \gamma^t r(s_t, a_t)
$$

于是：

$$
J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
R(\tau)
\right]
$$

轨迹概率为：

$$
p_\theta(\tau)
=
\rho(s_0)
\prod_{t=0}^{T}
\pi_\theta(a_t | s_t)
P(s_{t+1} | s_t, a_t)
$$

其中环境转移概率 \(P\) 不依赖 \(\theta\)，策略 \(\pi_\theta\) 依赖 \(\theta\)。

## 1. Policy Gradient

我们想求：

$$
\nabla_\theta J(\theta)
$$

直接对期望求导：

$$
\nabla_\theta J(\theta)
=
\nabla_\theta
\int p_\theta(\tau) R(\tau) d\tau
$$

$$
=
\int
\nabla_\theta p_\theta(\tau)
R(\tau)
d\tau
$$

使用 log-derivative trick：

$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta \log p_\theta(\tau)
$$

所以：

$$
\nabla_\theta J(\theta)
=
\int
p_\theta(\tau)
\nabla_\theta \log p_\theta(\tau)
R(\tau)
d\tau
$$

也就是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log p_\theta(\tau)
R(\tau)
\right]
$$

而：

$$
\log p_\theta(\tau)
=
\log \rho(s_0)
+
\sum_{t=0}^{T}
\log \pi_\theta(a_t | s_t)
+
\sum_{t=0}^{T}
\log P(s_{t+1}|s_t,a_t)
$$

只有策略项依赖 \(\theta\)，因此：

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t | s_t)
$$

于是得到经典 Policy Gradient：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t | s_t)
R(\tau)
\right]
$$

这就是 REINFORCE 的基本形式。

不过这里使用整条轨迹奖励 \(R(\tau)\)，方差通常很大。更常见的是使用从当前时刻开始的 return：

$$
G_t
=
\sum_{k=t}^{T}
\gamma^{k-t} r(s_k, a_k)
$$

于是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t | s_t)
G_t
\right]
$$

进一步，可以减去 baseline \(b(s_t)\)，且不改变期望：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t | s_t)
\left(G_t - b(s_t)\right)
\right]
$$

通常取：

$$
b(s_t) = V^\pi(s_t)
$$

定义 advantage：

$$
A^\pi(s_t, a_t)
=
Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

最终常用形式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t | s_t)
A^\pi(s_t, a_t)
\right]
$$

直觉：

```text
如果 A(s, a) > 0，说明这个动作比平均水平好，提高它的概率。
如果 A(s, a) < 0，说明这个动作比平均水平差，降低它的概率。
```

## 2. Actor-Critic

Policy Gradient 只训练策略，方差较大。Actor-Critic 同时训练两个模型：

```text
Actor:  pi_theta(a | s)，负责选动作
Critic: V_phi(s)，负责估计状态价值
```

优势函数可以用 TD error 近似：

$$
\delta_t
=
r_t
+
\gamma V_\phi(s_{t+1})
-
V_\phi(s_t)
$$

于是 actor 更新：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t | s_t)
\delta_t
\right]
$$

critic 更新则最小化：

$$
L_V(\phi)
=
\left(
V_\phi(s_t)
-
\hat{G}_t
\right)^2
$$

或者 TD loss：

$$
L_V(\phi)
=
\left(
V_\phi(s_t)
-
\left(r_t + \gamma V_\phi(s_{t+1})\right)
\right)^2
$$

在大语言模型 RLHF 中，actor 是语言模型，critic/value model 估计某个 token 或 response 的价值。

## 3. TRPO

Policy Gradient 有一个问题：一步更新太大可能让策略崩掉。

TRPO，Trust Region Policy Optimization，核心思想是：

```text
最大化策略改进目标，
同时限制新旧策略之间的 KL divergence。
```

TRPO 目标来自 importance sampling。定义概率比：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t | s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t | s_t)
}
$$

surrogate objective 为：

$$
L(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta) A_t
\right]
$$

如果 \(A_t > 0\)，希望提高该动作概率；如果 \(A_t < 0\)，希望降低该动作概率。

为了避免变化过大，TRPO 加 KL 约束：

$$
\max_\theta
\mathbb{E}_t
\left[
r_t(\theta) A_t
\right]
$$

subject to:

$$
\mathbb{E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot|s_t)
\|
\pi_\theta(\cdot|s_t)
\right)
\right]
\leq \delta
$$

推导上，TRPO 使用局部二阶近似。目标函数一阶近似：

$$
L(\theta)
\approx
L(\theta_{\mathrm{old}})
+
g^T(\theta - \theta_{\mathrm{old}})
$$

KL 约束二阶近似：

$$
D_{\mathrm{KL}}
\approx
\frac{1}{2}
(\theta - \theta_{\mathrm{old}})^T
H
(\theta - \theta_{\mathrm{old}})
$$

其中 \(H\) 是 KL 的 Hessian，也对应 Fisher Information Matrix。

令：

$$
x = \theta - \theta_{\mathrm{old}}
$$

近似优化问题变成：

$$
\max_x
g^T x
$$

subject to:

$$
\frac{1}{2}x^T Hx \leq \delta
$$

其解方向是 natural gradient：

$$
x
\propto
H^{-1} g
$$

最终步长为：

$$
x
=
\sqrt{
\frac{2\delta}
{g^T H^{-1}g}
}
H^{-1}g
$$

TRPO 理论上很漂亮，但工程复杂，需要 conjugate gradient、line search、Hessian-vector product 等。

## 4. PPO

PPO，Proximal Policy Optimization，是 TRPO 的简化版。

它同样使用概率比：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t | s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t | s_t)
}
$$

朴素 surrogate objective：

$$
L^{PG}(\theta)
=
\mathbb{E}_t
\left[
r_t(\theta) A_t
\right]
$$

但如果 \(r_t(\theta)\) 远离 1，说明新策略和旧策略差异太大。

PPO 使用 clipped objective：

$$
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\operatorname{clip}
\left(
r_t(\theta),
1-\epsilon,
1+\epsilon
\right)
A_t
\right)
\right]
$$

分情况看：

如果 \(A_t > 0\)，说明动作好，想增大概率，也就是希望 \(r_t > 1\)。但 PPO 不允许增大太多：

$$
r_t(\theta)
\leq 1+\epsilon
$$

目标变成：

$$
\min
\left(
r_t A_t,
(1+\epsilon)A_t
\right)
$$

当 \(r_t > 1+\epsilon\) 后，收益不再增加。

如果 \(A_t < 0\)，说明动作差，想降低概率，也就是希望 \(r_t < 1\)。但 PPO 不允许降低太多：

$$
r_t(\theta)
\geq 1-\epsilon
$$

直觉：

```text
好动作可以提高概率，但不能提高太多。
坏动作可以降低概率，但不能降低太多。
```

PPO 的完整 loss 通常包含三项：

$$
L(\theta, \phi)
=
L^{CLIP}(\theta)
-
c_1 L^{VF}(\phi)
+
c_2 H(\pi_\theta)
$$

其中：

```text
L^{CLIP}: policy loss
L^{VF}: value function loss
H: entropy bonus，鼓励探索
```

很多实现采用最小化 loss，所以会写成：

$$
\mathcal{L}_{policy}
=
-
L^{CLIP}
$$

在 RLHF 中，PPO 通常还会加入 KL penalty，限制当前模型不要偏离 reference model 太远：

$$
r_{\mathrm{total}}
=
r_{\mathrm{reward\ model}}
-
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta
\|
\pi_{\mathrm{ref}}
\right)
$$

token-level 可以写成：

$$
r_t
=
-
\beta
\left(
\log \pi_\theta(y_t|x,y_{<t})
-
\log \pi_{\mathrm{ref}}(y_t|x,y_{<t})
\right)
$$

然后在最后一个 token 加上 reward model 给出的整体 response reward。

## 5. PPO 在 RLHF 中的流程

以语言模型为例，prompt 是 \(x\)，模型回答是 \(y\)。

流程：

```text
1. 从 prompt dataset 采样 x
2. 当前 policy pi_theta 生成回答 y
3. reward model 给回答打分 r(x, y)
4. 加 KL penalty，防止偏离 reference model
5. 用 value model 估计 advantage
6. 用 PPO 更新 policy
```

目标大致是：

$$
\max_\theta
\mathbb{E}_{x, y \sim \pi_\theta}
\left[
r_\psi(x,y)
-
\beta
\log
\frac{
\pi_\theta(y|x)
}{
\pi_{\mathrm{ref}}(y|x)
}
\right]
$$

PPO 优点：

```text
稳定
实现比 TRPO 简单
可以复用一批 rollout 做多轮 minibatch update
```

PPO 缺点：

```text
需要 reward model
需要 value model
训练复杂，超参数敏感
容易 reward hacking
显存和工程成本高
```

## 6. DPO

DPO，Direct Preference Optimization，是 RLHF 里的重要替代方案。

它不显式训练 reward model，也不跑 PPO，而是直接用偏好数据优化模型。

偏好数据形式：

```text
prompt: x
chosen response: y_w
rejected response: y_l
```

意思是人类更喜欢 \(y_w\) 而不是 \(y_l\)。

DPO 的目标来自 KL-regularized RL：

$$
\max_\pi
\mathbb{E}_{x, y \sim \pi}
\left[
r(x,y)
\right]
-
\beta
D_{\mathrm{KL}}
\left(
\pi(y|x)
\|
\pi_{\mathrm{ref}}(y|x)
\right)
$$

对固定 \(x\)，目标是：

$$
\max_{\pi}
\sum_y
\pi(y|x) r(x,y)
-
\beta
\sum_y
\pi(y|x)
\log
\frac{
\pi(y|x)
}{
\pi_{\mathrm{ref}}(y|x)
}
$$

这个优化问题的最优解满足：

$$
\pi^*(y|x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y|x)
\exp
\left(
\frac{1}{\beta} r(x,y)
\right)
$$

等价地：

$$
r(x,y)
=
\beta
\log
\frac{
\pi^*(y|x)
}{
\pi_{\mathrm{ref}}(y|x)
}
+
\beta \log Z(x)
$$

因为 \(\log Z(x)\) 只依赖 \(x\)，比较同一个 prompt 下两个 response 时会抵消：

$$
r(x,y_w) - r(x,y_l)
=
\beta
\left[
\log
\frac{
\pi^*(y_w|x)
}{
\pi_{\mathrm{ref}}(y_w|x)
}
-
\log
\frac{
\pi^*(y_l|x)
}{
\pi_{\mathrm{ref}}(y_l|x)
}
\right]
$$

偏好模型一般使用 Bradley-Terry model：

$$
P(y_w \succ y_l | x)
=
\sigma
\left(
r(x,y_w) - r(x,y_l)
\right)
$$

把 reward 差代入，并用当前 policy \(\pi_\theta\) 近似 \(\pi^*\)，得到：

$$
P_\theta(y_w \succ y_l | x)
=
\sigma
\left(
\beta
\left[
\log
\frac{
\pi_\theta(y_w|x)
}{
\pi_{\mathrm{ref}}(y_w|x)
}
-
\log
\frac{
\pi_\theta(y_l|x)
}{
\pi_{\mathrm{ref}}(y_l|x)
}
\right]
\right)
$$

最大化偏好 likelihood，等价于最小化 DPO loss：

$$
\mathcal{L}_{DPO}(\theta)
=
-
\mathbb{E}_{(x,y_w,y_l)}
\left[
\log
\sigma
\left(
\beta
\left[
\log
\frac{
\pi_\theta(y_w|x)
}{
\pi_{\mathrm{ref}}(y_w|x)
}
-
\log
\frac{
\pi_\theta(y_l|x)
}{
\pi_{\mathrm{ref}}(y_l|x)
}
\right]
\right)
\right]
$$

展开成 logprob：

$$
\mathcal{L}_{DPO}
=
-
\log
\sigma
\left(
\beta
\left[
\log \pi_\theta(y_w|x)
-
\log \pi_\theta(y_l|x)
-
\log \pi_{\mathrm{ref}}(y_w|x)
+
\log \pi_{\mathrm{ref}}(y_l|x)
\right]
\right)
$$

直觉：

```text
让当前模型相对于 reference model 更偏向 chosen，
更不偏向 rejected。
```

DPO 优点：

```text
不需要训练 reward model
不需要 PPO rollout
不需要 value model
训练像普通 supervised fine-tuning 一样简单
稳定性通常更好
```

DPO 缺点：

```text
依赖 pairwise preference 数据
本质是离线优化，不能像 PPO 那样主动探索
对数据质量敏感
beta 影响很大
```

## 7. DPO 和 SFT 的区别

SFT 只最大化 chosen response：

$$
\mathcal{L}_{SFT}
=
-
\log \pi_\theta(y_w | x)
$$

DPO 同时看 chosen 和 rejected：

$$
\mathcal{L}_{DPO}
=
-
\log
\sigma
\left(
\beta
\left[
\Delta_\theta - \Delta_{\mathrm{ref}}
\right]
\right)
$$

其中：

$$
\Delta_\theta
=
\log \pi_\theta(y_w|x)
-
\log \pi_\theta(y_l|x)
$$

$$
\Delta_{\mathrm{ref}}
=
\log \pi_{\mathrm{ref}}(y_w|x)
-
\log \pi_{\mathrm{ref}}(y_l|x)
$$

所以 DPO 不是简单地让模型学 chosen，而是让模型学：

```text
chosen 相比 rejected 应该更被偏好，
而且这个偏好强度是相对于 reference model 来说的。
```

## 8. GRPO

GRPO，Group Relative Policy Optimization，常见于大模型 reasoning 训练场景。

它和 PPO 很像，但一个重要区别是：

```text
PPO 通常需要 value model 估计 advantage。
GRPO 不训练 value model，而是在同一个 prompt 下采样多个 responses，用组内 reward 做相对优势。
```

设对同一个 prompt \(q\)，当前 policy 采样 \(G\) 个回答：

$$
\{o_1, o_2, ..., o_G\}
$$

每个回答得到 reward：

$$
r_i = r(q, o_i)
$$

然后计算组内均值和标准差：

$$
\mu
=
\frac{1}{G}
\sum_{i=1}^{G}
r_i
$$

$$
\sigma
=
\sqrt{
\frac{1}{G}
\sum_{i=1}^{G}
(r_i-\mu)^2
}
$$

定义相对优势：

$$
A_i
=
\frac{
r_i - \mu
}{
\sigma + \epsilon
}
$$

对第 \(i\) 个回答的第 \(t\) 个 token，概率比为：

$$
r_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}|q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})
}
$$

GRPO 的 clipped objective 类似 PPO：

$$
\mathcal{J}_{GRPO}(\theta)
=
\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\min
\left(
r_{i,t}(\theta) A_i,
\operatorname{clip}
\left(
r_{i,t}(\theta),
1-\epsilon,
1+\epsilon
\right)
A_i
\right)
\right]
$$

通常还会加 KL penalty，限制当前 policy 偏离 reference model：

$$
-
\beta
D_{\mathrm{KL}}
\left(
\pi_\theta
\|
\pi_{\mathrm{ref}}
\right)
$$

所以完整目标可写成：

$$
\mathcal{J}_{GRPO}(\theta)
=
\mathbb{E}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left\{
\min
\left(
r_{i,t} A_i,
\operatorname{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i
\right)
-
\beta
D_{\mathrm{KL}}
\left[
\pi_\theta
\|
\pi_{\mathrm{ref}}
\right]
\right\}
\right]
$$

直觉：

```text
对同一个问题生成多个答案。
谁的 reward 高于组内平均，谁的 token 概率就被提高。
谁的 reward 低于组内平均，谁的 token 概率就被降低。
```

这相当于用组内均值作为 baseline：

$$
A_i = r_i - \mathrm{baseline}
$$

其中 baseline 不是由 value model 预测，而是直接由同一个 prompt 的多个采样结果估计。

## 9. GRPO 为什么可以去掉 value model

在 PPO 中，优势函数通常是：

$$
A_t
=
Q(s_t,a_t) - V(s_t)
$$

这里 \(V(s_t)\) 是状态 baseline，用来降低方差。

在 LLM response-level RL 中，对同一个 prompt \(q\)，每个完整回答 \(o_i\) 有一个 reward \(r_i\)。如果采样多个回答，就可以用组内平均奖励近似 baseline：

$$
b(q)
\approx
\frac{1}{G}
\sum_i r_i
$$

于是：

$$
A_i
=
r_i - b(q)
$$

这就绕开了 value model。

优势：

```text
省掉 value model
省显存
训练流程更简单
适合可验证 reward 的 reasoning 任务
```

缺点：

```text
需要每个 prompt 采样多个答案
组大小 G 太小会导致 advantage 噪声大
如果 reward 稀疏，组内区分度可能不足
```

## 10. 几个算法的关系

```text
Policy Gradient:
  nabla log pi(a|s) * advantage

TRPO:
  Policy Gradient + KL trust region
  用硬约束限制新旧策略距离

PPO:
  TRPO 的简单近似
  用 clipped probability ratio 限制更新幅度

DPO:
  从 KL-regularized RL 推出
  不显式训练 reward model，不在线采样
  直接用 preference pair 更新 policy

GRPO:
  PPO-style RL
  不用 value model
  对同一 prompt 采样一组 responses
  用组内相对 reward 估计 advantage
```

## 11. 从目标函数角度比较

Policy Gradient：

$$
\nabla_\theta J
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a|s) A
\right]
$$

TRPO：

$$
\max_\theta
\mathbb{E}
\left[
r_t(\theta) A_t
\right]
$$

subject to:

$$
\mathbb{E}
\left[
D_{\mathrm{KL}}(\pi_{\mathrm{old}} \| \pi_\theta)
\right]
\leq \delta
$$

PPO：

$$
\max_\theta
\mathbb{E}
\left[
\min
\left(
r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

DPO：

$$
\min_\theta
-
\mathbb{E}
\left[
\log \sigma
\left(
\beta
\left[
\log
\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
-
\log
\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}
\right]
\right)
\right]
$$

GRPO：

$$
\max_\theta
\mathbb{E}
\left[
\frac{1}{G}
\sum_i
\frac{1}{|o_i|}
\sum_t
\min
\left(
r_{i,t} A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
-
\beta KL
\right]
$$

其中：

$$
A_i
=
\frac{
r_i - \operatorname{mean}(r_1,\dots,r_G)
}{
\operatorname{std}(r_1,\dots,r_G) + \epsilon
}
$$

## 12. 一句话理解

Policy Gradient：

```text
让高回报动作更可能出现，让低回报动作更不可能出现。
```

TRPO：

```text
每次只在可信区域内更新策略，避免策略崩掉。
```

PPO：

```text
用 clip 限制概率比，简单稳定地近似 TRPO。
```

DPO：

```text
不用 reward model 和 PPO，直接从 chosen/rejected 偏好对中优化 policy。
```

GRPO：

```text
对同一问题采样一组答案，用组内相对好坏作为 advantage，省掉 value model。
```

## 13. 在 LLM 对齐中的典型位置

```text
Pretraining
  -> 预测下一个 token

SFT
  -> 学人类/专家示范答案

Reward Modeling
  -> 用偏好数据训练 reward model

PPO RLHF
  -> 用 reward model 强化学习优化模型

DPO
  -> 跳过 reward model + PPO，直接用偏好数据优化

GRPO
  -> 用规则奖励/验证器奖励训练 reasoning，常用于数学、代码等可验证任务
```

核心差异可以压缩成：

```text
PPO/TRPO 是在线 RL，需要采样当前 policy 的输出。
DPO 是离线 preference learning，直接吃 chosen/rejected 数据。
PPO 需要 value model。
GRPO 通常不需要 value model，而用 group reward normalization。
TRPO 理论更干净，但工程更复杂。
PPO 是实践中更常用的稳定近似。
```
