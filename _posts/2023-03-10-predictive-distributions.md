---
layout: post
title: "On Evaluating Probablistic Predictions"
date: 2023-6-1
tags: machine-learning
---

> Point predictions are often unsufficient and for machine learning tasks that have to deal with uncertainty. Instead, probablistic models are becoming more widely used. This post talks about proper scoring rules, a framework to think about and evaluate probablistic predictions.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## What are Predictive Distributions?


I've been seeing a trend of using probablistic forecasts for machine learning tasks. It feels natural - in many real world situations there exist uncertainty that doesn't but requires a more expressive representation. With predictive distributions, we can have better uncertainty quantification, and make better decisions given these uncertainties. 

To put in perspective, a predictive distribution is when given input $$X$$, instead of outputing a point prediction $$y$$, the model outputs a distribution over the space of outputs $$p(y\vert X)$$.

For example, for a classifcation task, we can output confidence along with a label:

![]({{ '/assets/img/properscoring/classification.png' | relative_url }}){: style="width: 60%;" } 

For regression, one can output a gaussian distribution over $$y$$ (mean and variance), like in the case of Gaussian Process regression:

![]({{ '/assets/img/properscoring/gp.png' | relative_url }}){: style="width: 60%;" } 

Uncertainties can be from different sources. The most mainstream distinction is aleatoric (there exists variations in the data, as in our classification example) vs epstemic (we don't have enough data to know the label for sure, as in our regression example). 

![Illustration from TODO]({{ '/assets/img/properscoring/uq.png' | relative_url }}){: style="width: 90%;" } 
*<center>Illustration from TODO</center>*

### Calibration and Sharpness

In general, we want two thing from our probablistic forecasts: calibration, and accuracy. Calibration refers to that the probabilities predicted acctually corresponds to the true probablies. For example,  among the days that we predict 30% parcipitation rate, 30% of them actually rained.

A paper from [TODO] et al showed that as neural netowkrs get deeper, they are becoming less calibrated. It's an interesting finding and 



## Proper Scoring Rules

### Definition

Scoring rules are functions that score how well a predictive distribution captures data. Let's say we have data domain $$x \sim \mathbf{X}$$, and $$P$$ is a distirbution over $$\mathbf{X}$$. Then the score

$$S: P \times x \rightarrow \mathbb{R} $$

It assess the quality of probabilistic forecasts, by assigning a numerical score based on the predictive distribution and on the event or value that materializes. A scoring rule is proper if the forecaster maximizes the expected score for an observation drawn from the distribution $$F$$ if he or she issues the probabilistic forecast $$F$$, rather than $$G = F$$. It is strictly proper if the maximum is unique. 


### Some Examples

- Negative Log Likelyhood (NLL)

Perhaps the most well known of the proper scoring rules. 

- Brier/Quadratic Score

- Mean Interval Score

- Energy score 
Energy Score (ES) is a proper scoring rule to measure calibration and sharpness of the predicted distributions. Defined as $$\text{ES}(P, \textbf{x}) = E_{ \textbf{X} \sim P} \| \textbf{X} - \textbf{x} \| - \frac{1}{2} E_{ \textbf{X, X'} \sim P} \| \textbf{X} - \textbf{X'} \| $$

The energy score parameter-free measure, which is easy to use and implement. Recent [research](https://arxiv.org/pdf/2010.03759.pdf) shows that the energy score outperforms the softmax confidence score on common OOD evaluation benchmarks. (this is based on energy models not the same? check)

- KL divergence is not a proper scoring rule


### Is KL divergence a proper scoring rule?


Short answer: no. 

The Kullback-Leibler (KL) divergence, also known as relative entropy, is a measure of dissimilarity between two probability distributions. It is often used in information theory and machine learning to compare an estimated probability distribution with a true or reference distribution. However, KL divergence is not considered a proper scoring rule. Here's why:

1. Asymmetry: KL divergence is not symmetric, meaning that $$KL(P \vert Q) $$is not equal to $$KL(Q \vert P)$$ for probability distributions P and Q. This asymmetry implies that KL divergence doesn't treat the predicted and true distributions in the same manner. Proper scoring rules, on the other hand, are designed to assess the accuracy of forecasts in a symmetric way.

2. Not a direct measure of forecast accuracy: KL divergence measures the difference between two probability distributions but does not directly assess the accuracy of a single probabilistic forecast with respect to the true outcome. Proper scoring rules, such as Brier score or logarithmic score, are designed to evaluate the quality of probabilistic forecasts based on their correspondence to the actual outcomes.

3. Non-strictly proper: A scoring rule is considered strictly proper if the expected score is uniquely maximized when the predicted probability distribution is equal to the true distribution. While KL divergence is proper in the sense that it is minimized when the predicted distribution matches the true distribution, it is not strictly proper because it doesn't have a unique maximum. In other words, multiple predicted distributions can yield the same KL divergence value.
Because of these reasons, KL divergence does not qualify as a proper scoring rule. Instead, it serves as a useful measure of dissimilarity between probability distributions in various applications, such as model selection, optimization, and information theory.



-------------

_Extended Readings_

- [Strictly proper scoring rules, prediction, and estimation.](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf). Gneiting, Tilmann, and Adrian E. Raftery. The OG proper scoring rules paper. 

-  [Valid prediction intervals for regression problems](https://arxiv.org/pdf/2107.00363.pdf). Nicolas Dewolf, Bernard De Baets, Willem Waegeman. A survey of probablistic forecasting models including Bayesian methods (GP, BNN), ensemble methods, direct estimation methods (quantile regression), and conformal prediction methods. They introduce the methods clearly and compare the advantages and disandvantages of each. 

-------------

_References_


 Uncertainty in Deep Learning. How To Measure? [blogpost](https://towardsdatascience.com/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok-50ffa562cb0b), Michel Kana, 2020

Gneiting, Tilmann, and Adrian E. Raftery. "Strictly proper scoring rules, prediction, and estimation." [pdf](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)_Journal of the American statistical Association_ 102.477 (2007): 359-378.


Victor Richmond Jose, Robert Nau, & Robert Winkler. "Scoring Rules, Generalized Entropy, and Utility Maximization" [slides](https://people.duke.edu/~rnau/Nau_Scoring_Rules_Paris_seminar.pdf), Presentation for GRID/ENSAM Seminar, 2007

