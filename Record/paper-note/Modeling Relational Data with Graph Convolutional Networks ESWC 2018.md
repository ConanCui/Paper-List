# Modeling Relational Data with Graph Convolutional Networks

The author introdcued Relational Graph Convolutional Networks (RGCN), which is based on a recent class of neural networks operating on graphs. The RGCN is designed to handle highly multi-ralational data which is the main contribution as the author said, and perform well on two statistical relation learning (SRL) tasks link prediction and entity classification.

Here give the notation:

1. directed and labeled multi-graphs $G = ( \mathcal { V } , \mathcal { E } , \mathcal { R } )$
2. nodes entities $v _ { i } \in \mathcal { V }$
3. labeled edges  $\left( v _ { i } , r , v _ { j } \right) \in \mathcal { E }$
4. relation type $r \in \mathcal { R }$


## Relational Graph Convolutional Networks
Like other GCN that operate on local graph neighborhoods, we have following message-passing framwork on largel scale relational data:
$$h _ { i } ^ { ( l + 1 ) } = \sigma \left( \sum _ { m \in \mathcal { M } _ { i } } g _ { m } \left( h _ { i } ^ { ( l ) } , h _ { j } ^ { ( l ) } \right) \right)$$

where $h _ { i } ^ { ( l ) } \in \mathbb { R } ^ { d ^ { ( l ) } }$ is the hidden state of node $v_i$ in $l$th layer. $\mathcal{M_i}$ is the set of incoming edge of node $v_i$. $g_m()$ is the neural netwrok to compute the incoming messages, e.g. the linear transformation $g _ { m } \left( h _ { i } , h _ { j } \right) = W h _ { j }$.

Motivated by this, this paper proposed a more specific messages passing method as below:
$$h _ { i } ^ { ( l + 1 ) } = \sigma \left( \sum _ { r \in \mathcal { R } } \sum _ { j \in \mathcal { N } _ { i } ^ { r } } \frac { 1 } { c _ { i , r } } W _ { r } ^ { ( l ) } h _ { j } ^ { ( l ) } + W _ { 0 } ^ { ( l ) } h _ { i } ^ { ( l ) } \right)$$
where $\mathcal { N } _ { i } ^ { r }$ is the set of neighbor of node $i$ under relation $r \in \mathcal { R }$.

There are main two difference from regular GCNS:
1. The message is passing depend on the specific relation type.
2. add a self-connection as a special relation type.

This Modification would also introduce a central issue, the number of parameters would grow radily with the number of relation type. This woudl make the model verfitting. So the author introduce two kinds of regularization as below:
1. basic decomposition (weight sharing between different relation types): $W _ { r } ^ { ( l ) } = \sum _ { b = 1 } ^ { B } a _ { r b } ^ { ( l ) } V _ { b } ^ { ( l ) }$, where $V _ { b } ^ { ( l ) } \in \mathbb { R } ^ { d ^ { ( l + 1 ) } \times d ^ { ( l ) } }$. And the coffecient $a _ { r b } ^ { ( l ) }$ need learn depend on $r$.
2. block-diagonal decomposition (sparsity regularization): $W _ { r } ^ { ( l ) } = \bigoplus _ { b = 1 } ^ { B } Q _ { b r } ^ { ( l ) }$. In this setting, the $W _ { r } ^ { ( l ) }$ are block-diagonal matrices: $\operatorname { diag } \left( Q _ { 1 r } ^ { ( l ) } , \ldots , Q _ { B r } ^ { ( l ) } \right)$, $Q _ { b r } ^ { ( l ) } \in \mathbb { R } ^ { \left( d ^ { ( l + 1 ) } / B \right) \times \left( d ^ { ( l ) } / B \right) }$. And when $B=d$, $W_r$ becomes a diagonal matrix.

The overall R-GCN model would stack $L$ layers, notice that the input of the first layer take unique one-hot vector for each node if no feature are provided.

## Entity classification
In this task, the author just stack R-GCN layers with a softmax activation on the last layer. The model is trained following rhe multinomial likelihood as below:
$$\mathcal { L } = - \sum _ { i \in \mathcal { Y } } \sum _ { k = 1 } ^ { K } t _ { i k } \ln h _ { i k } ^ { ( L ) }$$

## Link prediction
In this setting, rather than the full set of edges $\mathcal { E }$, the model are given incomplete subsets $\hat { \varepsilon }$. And the task is to assing scores to possible edges in order to dertermine how likely yheose edges are belong to $\mathcal { E }$.

The author stack R-GCN with the DistMult factorization score function. Given the triplets (subject, relation, object), the model take this (\mathbb { R } ^ { d } \times \mathcal { R } \times \mathbb { R } ^ { d }) as input, and output a score as below:

$$f ( s , r , o ) = e _ { s } ^ { T } R _ { r } e _ { o }$$

The model is trained with negative sampling (one oberserved example with $w$ negative example). And the model optimize for cross-entropy loss as below:

$$\begin{array} { r l } { \mathcal { L } = - \frac { 1 } { ( 1 + \omega ) | \hat { \mathcal { E } } | } } & { \sum _ { ( s , r , o , y ) \in \mathcal { T } } y \log l ( f ( s , r , o ) ) + } \\ { } & { ( 1 - y ) \log ( 1 - l ( f ( s , r , o ) ) ) } \end{array}$$

## Experiment

