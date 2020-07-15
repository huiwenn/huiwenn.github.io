---
layout: post
title: "VC dimension, Rademacher Complexity, and localization"
tags: ml-theory
---

> In this post we'll talk about some complexity metrics for hypothesis classes in learning theory. 

<!--more-->


{: class="table-of-content"}
* TOC
{:toc}

## Why do we care?

One of the most insightful thing I learned last year was the theoretical framework of machine learning. For me, it provided a structured way to understand sample complexity (how many data we need to learn a classification/regression/policy), and when/why learning is effective in relation to data, algorithm, and hypothesis class. Hurray! One step away from alchemy.

## Syntax testing

$$
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha (r + \gamma \color{red}{\max_{a' \in \mathcal{A}} Q(s', a')})
$$
 
In a naive implementation, the Q value for all (s, a) pairs can be simply tracked in a dict. No complicated machine learning model is involved yet.
```python
from collections import defaultdict
Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param

env = gym.make("CartPole-v0")
actions = range(env.action_space)

def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s_next, a] for a in actions]) 
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])
```