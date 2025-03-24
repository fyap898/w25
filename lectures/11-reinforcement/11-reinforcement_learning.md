---
marp: true
paginate: true
theme: marp-theme
math: true
title: Reinforcement Learning
---

<!-- 
_class: invert lead
_paginate: skip
 -->

# (Deep) Reinforcement Learning

COMP 4630 | Winter 2025
Charlotte Curtis

---

## Overview

- Terminology and fundamentals
- Q-learning
- Deep Q Networks
- References and suggested reading:
    - [Scikit-learn book](https://librarysearch.mtroyal.ca/discovery/fulldisplay?context=L&vid=01MTROYAL_INST:02MTROYAL_INST&search_scope=MRULibrary&isFrbr=true&tab=MRULibraryResources&docid=alma9923265933604656): Chapter 18
    - [d2l.ai](https://d2l.ai/chapter_reinforcement-learning/index.html): Chapter 17

---

## Reinforcement Learning + LLMs
  ![h:500 center](https://cdn-lfs.hf.co/repos/32/7d/327d13dcdeb581193de91d7ac0e90f2ffc601e8c9acb8889a5aa64b42e504845/4a636e24cbce526aae13012e6813065b483c6dfd6ad9280f74ed1995454f5537?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27trl_overview.png%3B+filename%3D%22trl_overview.png%22%3B&response-content-type=image%2Fpng&Expires=1742792965&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0Mjc5Mjk2NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9yZXBvcy8zMi83ZC8zMjdkMTNkY2RlYjU4MTE5M2RlOTFkN2FjMGU5MGYyZmZjNjAxZThjOWFjYjg4ODlhNWFhNjRiNDJlNTA0ODQ1LzRhNjM2ZTI0Y2JjZTUyNmFhZTEzMDEyZTY4MTMwNjViNDgzYzZkZmQ2YWQ5MjgwZjc0ZWQxOTk1NDU0ZjU1Mzc%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=GvBuWky%7ExQlA8iLnoRPnKtifuBqp03pUACufJN3qE4lZpBVlRFthVJYj7hIIak3W0rMHjUF-25R7QvLAEwHFZC5luDUNyLaIuVxKLVMYRVQggMQJXblxL9C2Usi9wyF%7EZW7rix%7EY0Dq%7EMoQLSuUnG3QNou0hoMoctRDxKD2y%7E%7E3t73K7iKqO7TeyaU29XUp360G69AW5GiQAJojyYoXUcvw---z0AZ25txjJlekJVspwuYsSp6VVMfEspflEJAih2ulVuUArdRuXLUb5TmVe7JSo9BZLjqwwRb-SWgdBxP0Tsxc1MZAY80upJXqiK4iX9b9SwCZ4mpp7IOOXPadawg__&Key-Pair-Id=K3RPWS32NSSJCE)

<footer>Source: <a href="https://huggingface.co/blog/trl-peft">Hugging Face</a></footer>

---

## Terminology

- **Agent**: the learner or decision maker
- **Environment**: the world the agent interacts with
- **State**: the current situation
- **Reward**: feedback from the environment
- **Action**: what the agent can do
- **Policy**: the strategy the agent uses to make decisions

> Classic example: [Cartpole](https://jeffjar.me/cartpole.html)

---

## The Credit Assignment Problem

* Problem: If we've taken 100 actions and received a reward, which ones were "good" actions contributing to the reward?
* Solution: Evaluate an action based on the sum of all future rewards
    - Apply a **discount factor** $\gamma$ to future rewards, reducing their influence
    - Common choice in the range of $\gamma = 0.9$ to $\gamma = 0.99$
    - Example of actions/rewards:
        - Action: Right, Reward: 10
        - Action: Right, Reward: 0
        - Action: Right, Reward: -50

---

## Policy Gradient Approach

* If we can calculate the gradient of the **expected reward** with respect to the **policy parameters**, we can use gradient descent to find the best policy
* Approach:
    1. Play the game several times. At each step, compute the gradient (but don't update the policy yet).
    2. After several episodes, compute each action's **advantage** (relative sum of discounted rewards).
    3. Multiply each gradient vector by the advantage
    4. Compute the mean of all gradients and update the policy via gradient descent

<!-- Example in notebook -->

---

## Markov Chains

* A **Markov Chain** is a model of random states where the future state depends **only** on the current state (a **memoryless** process)
    ![center](../figures/11-fig18-7.png)
* Used to model real-world processes, e.g. Google's [PageRank algorithm](https://www.sciencedirect.com/science/article/pii/S016975529800110X?via%3Dihub)
* :question: Which of these is the **terminal state**?

<footer>Figure 18-7 from the Scikit-learn book</footer>

---

## Markov Decision Processes
![center](../figures/11-fig18-8.png)

* Like a Markov Chain, but with **actions** and **rewards**
* Bellman optimality equation:
    $$V^*(s) = \max_a \sum_{s'}T(s, a, s')[R(s, a, s') + \gamma V^*(s')] \text{ for all } s$$

---

## Iterative solution to Bellman's equation
**Value Iteration**:
1. Initialize $V(s) = 0$ for all states
2. Update $V(s)$ using the Bellman equation
3. Repeat until convergence

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'}T(s, a, s')[R(s, a, s') + \gamma V_k(s')] \text{ for all } s$$

> Problem: we still don't know the optimal policy

---

## Q-Values
Bellman's equation for Q-values (optimal state-action pairs):

$$Q_{k+1}(s, a) \leftarrow \sum_{s'}T(s, a, s')[R(s, a, s') + \gamma \max_{a'}Q_k(s', a')]$$

Optimal policy $\pi^*(s)$: 

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

For small spaces, we can use **dynamic programming** to iteratively solve for $Q^*$

---

## Q-Learning

* **Q-Learning** is a variation on Q-value iteration that learns the **transition probabilities** and **rewards** from experience
* An **agent** interacts with the environment and keeps track of the estimated Q-values for each state-action pair
* It's also a type of **temporal difference learning** (TD learning), which is kind of similar to stochastic gradient descent
* Interestingly, Q-learning is "off-policy" because it learns the optimal policy while following a different one (in this case, totally random exploration)

---

## Challenges with Q-Learning

* :question: We just converged on a 3-state problem in 10k iterations. How many states are in something like an Atari game?
* :question: How do we handle **continuous** state spaces?
* :question: How do you balance short-term rewards, long-term rewards, and exploration?

<div data-marpit-fragment>

One approach: **Approximate** Q-learning: 
* $Q_\theta(s, a)$ approximates the Q-value for any state-action pair
* The number of parameters $\theta$ can be kept manageable
* [Turns out](https://arxiv.org/abs/1312.5602) that **neural networks** are great for this!

</div>

---

## Deep Q-Networks
* We know states, actions, and observed rewards
* We need to estimate the Q-values for each state-action pair
* Target Q-values: $y(s, a) = r + \gamma \cdot \max_{a'}Q_\theta(s', a')$
    - $r$ is the observed reward, $s'$ is the next state
    - $Q_\theta(s', a')$ is the network's estimate of the future reward
* Loss function: $\mathcal{L}(\theta) = ||y(s, a) - Q_\theta(s, a)||^2$
* Standard MSE, backpropagation, etc.

---

## Challenges with DQNs
* **Catastrophic forgetting**: just when it seems to converge, the network forgets what it learned about old states and comes crashing down
* The **learning environment keeps changing**, which isn't great for gradient descent
* The **loss value** isn't a good indicator of performance, particularly since we're estimating both the target and the Q-values*
* Ultimately, reinforcement learning is inherently **unstable**!

---

<!-- 
_class: invert lead
_paginate: skip
 -->

 # The last topic: Geneterative AI and ethics