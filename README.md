# Optimizing Long-Term User Retention via Multi-Intervention Markov Decision Processes

## Overview

This project investigates how different app interventions affect long-term user retention using a Reinforcement Learning (RL) framework. Specifically, we model user engagement and churn as a **Markov Decision Process (MDP)** and train a **Deep Q-Network (DQN)** agent to learn optimal intervention strategies.

The study explores how app creators can balance engagement gains with user fatigue through interventions such as:

* Push notifications
* Feature additions
* Incentives and rewards
* Combined interventions
* No intervention

Results suggest that a **conservative policy emphasizing inaction** is often optimal, while stronger interventions should be reserved for users with low engagement or elevated churn risk.

---

## Motivation

Modern applications rely heavily on recommendation systems, notifications, and engagement algorithms to retain users. However, excessive interventions may lead to user fatigue and eventual churn.

This project asks:

> **How should app creators optimally intervene to maximize long-term retention while minimizing user fatigue?**

To answer this question, we formulate user retention as a sequential decision-making problem under uncertainty.

---

## Methodology

### Markov Decision Process (MDP)

The environment is modeled as an MDP consisting of five components:

1. Decision Epochs
2. State Space
3. Action Space
4. Reward Function
5. Transition Probabilities

The app creator acts as the **agent**, while the application ecosystem and users form the **environment**.

---

### Decision Epochs

* Finite horizon: **50 days**
* One time step corresponds to **one day**
* Simulates approximately **1–2 months of user activity**

---

### State Space

The user state is defined as:

[
s_t = (e_t, f_t, u_t, c_t)
]

where:

* (e_t \in [0,1]): user engagement
* (f_t \in [0,1]): user fatigue
* (u_t \in {\text{active, moderate, declining}}): user stage
* (c_t \in {0,1,2,3,4,5}): churn counter

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

* Engagement boost ((\alpha_a))
* Fatigue increase ((\delta_a))
* Intervention cost ((c(a)))

---

### Reward Function

The reward function is defined as:

[
r_t
===

## e_{t+1}

## \lambda f_{t+1}

## c(a_t)

\kappa \mathbf{1}_{\text{churn}}
]

where:

* Higher engagement increases reward
* Fatigue incurs penalties
* Interventions have operational costs
* Churn receives a large penalty ((\kappa = 7))

This reward structure encourages:

* Long-term retention
* User well-being
* Cost efficiency
* Strategic intervention timing

---

### Transition Dynamics

#### Engagement Transition

User engagement evolves according to:

[
e_{t+1} \sim \mathcal{N}(\mu_t,\sigma^2)
]

where

[
\mu_t
=====

e_t
+
\alpha_a(1-0.6f_t)
------------------

## \beta f_t^{1.05}

\eta(e_t-\bar e_u)
]

This incorporates:

* Positive intervention effects
* Fatigue penalties
* Mean reversion
* Random behavioral variation

---

#### Fatigue Transition

Fatigue evolves as:

[
f_{t+1}
=======

\min(1,\gamma f_t+\delta_a)
]

capturing:

* Fatigue accumulation from interventions
* Natural fatigue decay over time

---

#### Churn Dynamics

Users enter an absorbing churn state if engagement remains below threshold for five consecutive time steps:

[
c_t \ge 5
]

Once churn occurs:

* The episode terminates
* No additional rewards are received

---

## Deep Q-Network (DQN)

The MDP is solved using **Deep Q-Learning (DQN)**.

### Why DQN?

Traditional Q-learning becomes infeasible due to continuous state variables such as engagement and fatigue.

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

Input features:

* Engagement
* Fatigue
* Normalized churn risk
* One-hot encoded user stage

Output:

* Estimated Q-values for six actions

---

## Experimental Setup

### Training Parameters

| Parameter                  | Value   |
| -------------------------- | ------- |
| Episodes                   | 5,000   |
| Horizon                    | 50 days |
| Discount factor ((\gamma)) | 0.95    |
| Learning rate              | 0.001   |
| Batch size                 | 64      |
| Replay buffer              | 50,000  |
| Target update frequency    | 50      |

### Exploration

The agent follows an epsilon-greedy policy:

* Initial (\epsilon = 0.20)
* Minimum (\epsilon = 0.02)

---

## Ablation Studies

Four ablation studies were conducted:

1. **No Exploration** ((\epsilon = 0))
2. **Lower Churn Penalty** ((\kappa = 3))
3. **Higher Churn Penalty** ((\kappa = 12))
4. **Removal of Churn State**

These experiments evaluated the sensitivity of policy behavior to exploration and churn modeling.

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
* Conservative strategies outperform aggressive intervention policies.

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

Key references include:

* Sutton & Barto (2018), *Reinforcement Learning: An Introduction*
* Mnih et al. (2015), *Human-Level Control Through Deep Reinforcement Learning*
* Wang et al. (2021), *Long-Term Recommender Systems*
* Zhao et al. (2018), *Page-Wise Recommendations*
* O'Brien et al. (2022), *Push Notification Optimization*

---
