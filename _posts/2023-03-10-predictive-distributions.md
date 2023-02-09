---
layout: post
title: "On Evaluating Probablistic Predictions"
date: 2023-1-1
tags: machine-learning
---

> Point predictions are often unsufficient and for machine learning tasks that has to deal with uncertainty. This post talks about a frame work to think about and evaluate probablistic predictions, which includes proper scoring rules, multi-dimentional proper scoring rules, distribution-free uncertainty quantification, calibration vs. sharpness, and more.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Probablistic Predictions / Predictive Distributions

## Proper Scoring Rules

### Definition
Scoring rules assess the quality of probabilistic forecasts, by assigning a numerical score based on the predictive distribution and on the event or value that materializes. A scoring rule is proper if the forecaster maximizes the expected score for an observation drawn from the distribution $$F$$ if he or she issues the probabilistic forecast $$F$$, rather than $$G = F$$. It is strictly proper if the maximum is unique. 


### Some Examples

#### Brier/Quadratic Score

#### Mean Interval Score

#### KL divergence is not a proper scoring rule


## For multi-dimensional distributions, Proper Scoring Rules


## Distribution-free methods 

Conformal predictive distributions 

_References_

Gneiting, Tilmann, and Adrian E. Raftery. "Strictly proper scoring rules, prediction, and estimation." [pdf](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)_Journal of the American statistical Association_ 102.477 (2007): 359-378.


Victor Richmond Jose, Robert Nau, & Robert Winkler. "Scoring Rules, Generalized Entropy, and Utility Maximization" [slides](https://people.duke.edu/~rnau/Nau_Scoring_Rules_Paris_seminar.pdf), Presentation for GRID/ENSAM Seminar, 2007

