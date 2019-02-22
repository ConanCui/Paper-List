- [**SimplE Embedding for Link Prediction in Knowledge Graphs (NIPS 2013)**](./paper-note/Structural-Deep-Embedding-for-Hyper-Networks.md)
  - [Seyed Mehran Kazemi], David Poole
  - [Paper](https://arxiv.org/abs/1802.04868)
  - [tensorflow](https://github.com/Mehran-k/SimplE)
    This paper introduce a simple but effective method to learn effective embeddings for link prediction task. The model definition is as below. Firstly, the similarity function is defined as:

$$\frac { 1 } { 2 } \left( \left\langle h _ { e _ { i } } , v _ { r } , t _ { e _ { j } } \right\rangle + \left\langle h _ { e _ { j } } , v _ { r ^ { - 1 } } , t _ { e _ { i } } \right\rangle \right)$$
Then the model is trained through minimize the negative log-likelihood as below:
$$\min _ { \theta } \sum _ { ( ( h , r , t ) , l ) \in \mathrm { L } \mathrm { B } } \operatorname { softplus } ( - l \cdot \phi ( h , r , t ) ) + \lambda \| \theta \| _ { 2 } ^ { 2 }$$

This log-liklihood loss function is more robust to get overfitting compared to margin-based loss function which is used in Translational approaches.

This paper also has introduced other related work which can be categoriezed into
1. Translational approaches (TransE, TransH, TransR, STransE)
2. Multiplicative approaches (DistMult, ComplEx, RESCAL). Si
3. Deep Learning Approaches (E-MLP, ER-MLP)

And the author has discuss these models' restrictions in detail which give us the insight. Compared to these methods, the performance of SimplE is benefical from:
1. compared to DistMult, SimplE encode the asymmetric realationship into embedding.
  2.