# Optimizing Long-Term User Retention via Multi-Intervention Markov Decision Processes

## Overview

This project investigates how different app interventions affect long-term user retention using Reinforcement Learning (RL). User engagement and churn are modeled as a **Markov Decision Process (MDP)**, and a **Deep Q-Network (DQN)** is trained to learn optimal intervention strategies.

The intervention space includes:

* Push notifications
* Feature additions
* Incentives and rewards
* Combined interventions
* No intervention

Results indicate that a **conservative policy emphasizing inaction** is often optimal, while stronger interventions should be reserved for users with low engagement or elevated churn risk.

---

## Motivation

Modern applications rely heavily on recommendation systems, notifications, and engagement algorithms to retain users. However, excessive interventions may lead to user fatigue and eventual churn.

This project investigates the following research question:

> **How should app creators optimally intervene to maximize long-term retention while minimizing user fatigue?**

To answer this question, user retention is formulated as a sequential decision-making problem under uncertainty.

---

## Methodology

### Markov Decision Process (MDP)

The environment is modeled as an MDP consisting of five components:

1. Decision Epochs
2. State Space
3. Action Space
4. Reward Function
5. Transition Probabilities

The **agent** represents the app creator, while the **environment** consists of users and the application ecosystem.

---

### Decision Epochs

* Finite horizon: **50 days**
* One time step corresponds to **one day**
* Simulates approximately **1–2 months of user activity**

---

### State Space

The user state is defined as:

```
s_t = (e_t, f_t, u_t, c_t)
```

where:

* `e_t ∈ [0,1]`: user engagement
* `f_t ∈ [0,1]`: user fatigue
* `u_t ∈ {active, moderately engaged, declining}`: user stage
* `c_t ∈ {0,1,2,3,4,5}`: churn counter

---

### Action Space

The agent may select one of six interventions:

| Action | Description        |
| ------ | ------------------ |
| 0      | None               |
| 1      | Notify             |
| 2      | Feature            |
| 3      | Incentive          |
| 4      | Notify + Feature   |
| 5      | Notify + Incentive |

Each action is associated with:

* Engagement boost (`α_a`)
* Fatigue increase (`δ_a`)
* Intervention cost (`c(a)`)

---

### Reward Function

The reward function is:

```
r_t = e_(t+1) - λf_(t+1) - c(a_t) - κI_churn
```

where:

* Higher engagement increases reward
* Fatigue incurs penalties
* Interventions have operational costs
* Churn receives a large penalty (`κ = 7`)

This reward structure encourages:

* Long-term retention
* User well-being
* Cost efficiency
* Strategic intervention timing

---

### Transition Dynamics

#### Engagement Transition

User engagement evolves according to a Gaussian process:

```
e_(t+1) ~ N(μ_t, σ²)
```

with mean

```
μ_t = e_t + α_a(1 - 0.6f_t)
      - βf_t^1.05
      - η(e_t - ē_u)
```

This incorporates:

* Positive intervention effects
* Fatigue penalties
* Mean reversion
* Random behavioral variation

---

#### Fatigue Transition

Fatigue evolves according to:

```
f_(t+1) = min(1, γf_t + δ_a)
```

capturing:

* Fatigue accumulation from interventions
* Natural fatigue decay over time

---

#### Churn Dynamics

Users enter an absorbing churn state when engagement remains below threshold for five consecutive time steps:

```
c_t ≥ 5
```

Once churn occurs:

* The episode terminates
* No additional rewards are received

---

## Deep Q-Network (DQN)

The MDP is solved using **Deep Q-Learning (DQN)**.

### Why DQN?

Traditional tabular Q-learning becomes infeasible due to continuous state variables such as engagement and fatigue.

DQN enables:

* Function approximation
* Generalization to unseen states
* Scalable learning in large state spaces

---

### DQN Architecture

* Two hidden layers
* 64 neurons per layer
* ReLU activation
* Adam optimizer
* Experience replay buffer
* Target network updates

**Input features:**

* Engagement
* Fatigue
* Normalized churn risk
* One-hot encoded user stage

**Output:**

* Estimated Q-values for six actions

---

## Experimental Setup

### Training Parameters

| Parameter               | Value   |
| ----------------------- | ------- |
| Episodes                | 5,000   |
| Horizon                 | 50 days |
| Discount factor (γ)     | 0.95    |
| Learning rate           | 0.001   |
| Batch size              | 64      |
| Replay buffer           | 50,000  |
| Target update frequency | 50      |

### Exploration Policy

The agent follows an epsilon-greedy strategy:

* Initial ε = 0.20
* Minimum ε = 0.02

---

## Ablation Studies

Four ablation studies were conducted:

1. **No Exploration** (`ε = 0`)
2. **Lower Churn Penalty** (`κ = 3`)
3. **Higher Churn Penalty** (`κ = 12`)
4. **Removal of Churn State**

These experiments evaluated the sensitivity of learned policies to exploration and churn modeling.

---

## Key Results

* Final retention rate: **90.64%**
* Churn rate: **9.36%**
* Dominant action: **No intervention (89.76%)**
* Fatigue decreased over time
* Engagement gradually increased

The learned policy favored:

* Sparse interventions
* Fatigue-aware decision making
* Long-term optimization over short-term engagement gains

---

## Main Findings

* Frequent interventions are not always optimal.
* User fatigue plays a critical role in retention.
* Moderate churn penalties produce the best balance.
* Exploration is necessary to avoid suboptimal policies.
* Conservative strategies outperform aggressive intervention strategies.

---

## Limitations

* Simulated user data rather than real-world data
* Simplified engagement and fatigue dynamics
* Limited demographic and behavioral features

Future work may incorporate:

* Real app usage data
* Personalized intervention strategies
* Richer user representations
* Multi-agent or contextual RL methods

---

## References

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.*
* Mnih, V. et al. (2015). *Human-Level Control Through Deep Reinforcement Learning.*
* Wang, J. et al. (2021). *A Deep Reinforcement Learning Based Long-Term Recommender System.*
* Zhao, X. et al. (2018). *Deep Reinforcement Learning for Page-Wise Recommendations.*
* O'Brien, C. et al. (2022). *Optimizing Push Notification Decision Making by Modeling the Future.*

---
