# DReaM
This is an implementation of Discriminative Rectangle Mixture Model (DReaM). The model is described in [1]. 

[1] Junxiang Chen, Yale Chang, Brian Hobbs, Peter Castaldi, Michael Cho, Edwin Silverman, and Jennifer Dy "Interpretable Clustering via Discriminative Rectangle Mixture Model", Proceedings of the 16th IEEE International Conference on Data Mining (ICDM 2016), Barcelona, Spain, 2016

In the paper, we propose to apply variational inference to train the model. However, we discover that training the model with Expectation-Maximization (EM) is more efficient. Therefore, we provide the implementation of the EM algorithm.

# Example
We include examples in Demo.ipynb.
