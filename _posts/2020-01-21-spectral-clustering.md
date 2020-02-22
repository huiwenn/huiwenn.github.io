---
layout: post
title: "Implementing Self-Tuning Spectral Clustering"
date: 2020-01-21 01:09:00
tags: ml
---

> An (attempt at) implementation of the self-tuning spectral clustering algorithm by Zelnik-Manor and Perona (2014).

<!--more-->


{: class="table-of-content"}
* TOC
{:toc}


This implementation was a homework for a class called _Geometry of Data_ (very cool) taught by [Gal Mishne](http://mishne.ucsd.edu) (also very cool). 

One thing about CS grad school is that I started encountering problems whose solutions can't be easily found online or in books (how inconvenient!). This project, though, was thoroughly delightful. 


## Spectral Clustering

Spectral clustering is a approach to clustering where we (1) construct a graph from data and then (2) partition the graph by analyzing its connectivity.

This is a departure from some of the more well-known approaches, such as the K-means algorithm or learning a mixture model via EM, which are based on assumptions about how data are organized into clusters. They tend not to do the best when the clusters are of complex or unknown shape, for example:

(half moons)
(background)


## Implementing the NJW algorithm

Here we follow Ng, Jordan, and Weiss's algorithm [\[2\]](#references) for (not self-tuning) spectral clustering and implement it step by step. It goes like this:

Given a set of points $$S= \{s_1, \ldots, s_n\}$$ in $$\mathbb{R}^l$$ and desired number of clusters $$k$$:

1. Calculate affinity matrix $$A \in \mathbb{R}^{n \times n}$$ defined by 

    $$A_{ij} = \begin{cases} \exp (- \| s_i - s_j \| ^2 / 2 \sigma^2) & \text{for } i \neq j\\  0 & \text{for } i=j \end{cases}$$

    ```python
    import numpy as np
    from scipy.spatial.distance import pdist #Calculates pairwise distance

    def make_A(X, sigma):
        dim = X.shape[0]
        A = np.zeros([dim, dim])
        dist = iter(pdist(X))
        for i in range(dim):
            for j in range(i+1, dim):  
                d = np.exp(-1*next(dist)**2/(sigma**2))
                A[i,j] = d
                A[j,i] = d
        return A
    ```

2. Define $$D$$ to be the diagonal matrix whose $$(i,i)$$ element is the sum of $$A$$'s $$i$$-th row, and construct the Laplacian $$L = D^{-1/2}A D^{-1/2}$$.

    ```python
    from scipy import linalg

    A = make_A(data, 0.3) #Hyper parameter for sigma
    D = np.sum(A, axis=0) * np.eye(X.shape[0])
    D_half = linalg.fractional_matrix_power(D, -0.5)
    L = np.matmul(np.matmul(D_half, A), D_half)
    ```

3. Find the $$k$$ largest distinct eigenvectors $$x_1, x_2 , \ldots, x_k$$ and form the matrix $$X = [x_1x_2\dots x_k]$$ by stacking them together as columns.

    ```python
    # numpy returns the eigenvectors as a matrix where the 
    # i-th column corresponds to the i-th eigenvalue.
    # In the case of all real eigenvalues, they are organized from
    # small to large. So we are taking the slice of the last k columns.
    eigval, eigvec = np.linalg.eigh(L) 
    X = eigvec[:, -1*k:]
    ```
4. Re-normalize each row of $$X$$ to have unit length. Denote $$Y_{ij} = X_{ij}/(\sum_j X_{ij}^2)^{1/2}$$. This step is algorithmically cosmetic and deals with difference of connectivity within a cluster; one may skip this step if that gives better results empirically. 
    ```python
    row_sums = Y.sum(axis=1)
    Y = Y / row_sums[:, np.newaxis]
    ```

5. Cluster each row of $$Y$$ into $$k$$ clusters by K-means or any other algorithm. 
    ```python
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=k).fit(Y).labels_
    ```
6. Assign the original point $$s_i$$ to the cluster $$j$$ where the $$i$$-th row of matrix $$Y$$ was assigned to. Implementation-wise, we reuse the ```clusters``` variable directly.

To finish up, let's plot the results to see how it does - 

```python
import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1], c = clusters)
plt.title("Spectral Clustering")
```

## Tunning Oneself

### Local Scaling
The main drawback of the global $$\sigma$$ hyperparameter is that it does not handle the case where data is of various density across the domain, for which a $$\sigma$$ may work well with some portion of the data but not the rest. (see figure TODO for example.)

Zelnik-Manor and Perona \[1\] argues that connectivity/affinity (matrix $$A$$) should be viewed from the points themselves. We can formulate it by selecting a local scaling parameter for each data point $$s_i$$.
The distance from $$s_i$$ to $$s_j$$ from the perspective of $$s_i$$ is $$d(s_i , s_j )/\sigma_i$$ and the converse is  $$d(s_i , s_j )/\sigma_j$$. The square distance may be generalized as their product, i.e. $$d(s_i , s_j )^2/\sigma_i \sigma_j$$. The (local-scaled) affinity between a pair of points therefore becomes:

$$\hat{A}_{ij} = \exp(\frac{-d(s_i,s_j)^2}{\sigma_i\sigma_j})$$

Having such construct allows us to adapt the distance -> affinity scaling to data's local landscape. One way of achieving this goal, as introduced in \[1\], is to use the local nearest neighbor statistics 

$$\sigma_i = d(s_i, s_K)$$

Where $$s_K$$ is the $$K$$'th neighbor of $$s_i$$. So instead of a graph whose edge weight is absolute distance, we want to construct a weighted nearest neighbor graph. Hyperparameter $$K$$ will need to be manually tuned (\[1\] states that $$K=7$$ worked well for all their experiments).

Take data in the following figure for example. The local-scaled affinity will be high between points that are _both_ close neighbors to each other, whereby strongly connecting the red points as in (c) rather than in (b) where a single global $$\sigma$$ is used.

![figure 2]({{ '/assets/img/clustering/local-scaling.png' | relative_url }})
*Effect of local scaling, figure 2 from \[1\]. (a) input data point (b) affinity unscaled, and (c) affinity after local scaling*
{: style="width: 80%;" class="center"} 

The implementation of self tuning is straightforward. 

```python
from tqdm import tqdm

def W_local(X):
    dim = X.shape[0]
    dist_ = pdist(X)
    pd = np.zeros([dim, dim])
    dist = iter(dist_)
    for i in range(dim):
        for j in range(i+1, dim):  
            d = next(dist)
            pd[i,j] = d
            pd[j,i] = d
            
    #calculate local sigma
    sigmas = np.zeros(dim)
    for i in tqdm(range(len(pd))):
        sigmas[i] = sorted(pd[i])[7]
    
    A = np.zeros([dim, dim])
    dist = iter(dist_)
    for i in tqdm(range(dim)):
        for j in range(i+1, dim):  
            d = np.exp(-1*next(dist)**2/(sigmas[i]*sigmas[j]))
            #print(d)
            A[i,j] = d
            A[j,i] = d
    return A
```
## Syntax testing

$$
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha (r + \gamma \color{red}{\max_{a' \in \mathcal{A}} Q(s', a')})
$$
 

## References

[1] Lihi Zelnik-Manor, Pietro Perona. ["Self-Tuning Spectral Clustering"](https://arxiv.org/abs/1902.01342) NeurIPS 2014.

[2] Andrew Ng, Michael Jordan and Yair Weiss. ["On spectral clustering: Analysis and an algorithm"](https://www.semanticscholar.org/paper/On-Spectral-Clustering%3A-Analysis-and-an-algorithm-Ng-Jordan/c02dfd94b11933093c797c362e2f8f6a3b9b8012) NeurIPS 2001.

[3] Carl Doersch, Abhinav Gupta, and Alexei A. Efros. ["Unsupervised visual representation learning by context prediction."](https://arxiv.org/abs/1505.05192) ICCV. 2015.
