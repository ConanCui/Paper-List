# Awesome paper list

[![PRs Welcome](ReadMe.assets/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

A collection of graph embedding, deep learning, recommendation, knowledge graph, heterogeneous graph papers with reference implementations

<p align="center">
  <img width="460" src="Word Art.png">
</p>


<p align="center">
  <img width="1024" src="./ReadMe.assets/Graph convolutional neura.png">
</p>

You need to zoom in on this image to view this picture.

**Table of Contents**

1. [Recomendation](##Recomendation)
2. [Graph](##Graph)
3. [Bayesian Deep Learning](##BayesianDeepLearning)
4. [Datasets](##Datasets)



## daily read












| 日期 | 文章题目                                                     | 作者     |          会议          |                             代码                             | 备注                                                         |
| :--: | :----------------------------------------------------------- | -------- | :--------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|      | Relation-Aware Graph Convolutional Networks for Agent-Initiated Social E-Commerce Recommendation |          |       CIMK 2019        |               https://github.com/xfl15/RecoGCN               |                                                              |
|      | DEEP GAUSSIAN EMBEDDING OF GRAPHS:<br/>UNSUPERVISED INDUCTIVE LEARNING VIA RANKING |          |       ICLR 2018        |           https://www.kdd.in.tum.de/research/g2g/            | 提出概率建模，variance对应有具体的含义，另外提出ranking，即1 hop node embedding的similarity要大于2 hop node embedding，利用KL来计算similarity。同时，相比于node2vec这样的node embedding算法，该算法能够利用node attribute做到inductive，相比于graph sage，能够做到在test阶段，即使没有link，也能够产生node 的 embedding。 |
|      | Robust Graph Convolutional Networks Against Adversarial<br/>Attacks |          |        KDD 2019        | GCN: https://github.com/tkipf/gcn<br/>• GAT: https://github.com/PetarV-/GAT<br/>• RL-S2V: https://github.com/Hanjun-Dai/graph_adversarial_attack<br/>• NETTACK: https://github.com/danielzuegner/nettack | gcn中每一层特征都用一个gaussian distribution来表征，分布的好处是可以吸收对抗攻击的坏处。另外，设计了基于variance的attention方式，这样可以阻止对抗攻击在graph中的传播。 |
|      | Heterogeneous Graph Neural Network                           |          |        KDD 2019        |       https://github.com/chuxuzhang/KDD2019_HetGNN.git       | 相比于WWW中的GAT，多考虑了content的heterogeneous特性。       |
|      | Characterize and Transfer Attention in Graph Neural Networks |          | ICLR 2020 under review |                                                              | GAT在citation数据集上不同node的attention区分度不明显，在PPI上明显。这个attention和不同的数据集有着相关性，利用attention score作为feature vector，可以明显的区分出来不同的dataset。另外，作者尝试利用GAT得到的attention score对edge进行过滤，发现graph中的仅仅保留30-40%边仍能够得到不错的效果。 |
|      | Strategies for Pre-training Graph Neural Networks            |          | ICLR 2020 under review |                                                              | 1. node level self-supervised learning including masking attribute, structure similarity. 2. graph level supervised  multitask learning |
|      | Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View |          | AAAI 2020 under review |                                                              | 作者发现对于node classification这样的任务，inter class edge是有用的，而intra classi的edge是noisy的。作者提供了两种指标来衡量smoothing。同时作者还提出了两种方法来解决oversmooting，一种是加regularizer，在graph较近的node之间的feature vector的cosine distance变小，而graph上离得比较远的node之间的distance变大，另外一种方法为对graph进行重建，期望graph之间confidence比较高的edge得以保留，confidence比较低的边去除掉。这两种方法来使得在达到较高的层数的时候，performance的衰退变慢。 |
|      | PairNorm  Tackling Oversmoothing in GNNs                     |          | ICLR 2020 under review |                                                              |                                                              |
|      | Revisiting Semi-Supervised Learning with Graph Embeddings    |          |       ICML 2016        |                                                              | ![1571274794606](C:\Users\kncue\AppData\Roaming\Typora\typora-user-images\1571274794606.png) |
|      | 2020 - ICLR - DropEdge Towards Deep Graph Convolutional Networks on Node Classification |          | ICLR 2020 under review |                                                              |                                                              |
|      | Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks |          |        KDD 2019        |                                                              | <https://www.paperweekly.site/papers/3251>                   |
|      | Learning Discrete Structures for Graph Neural Networks       |          |       ICML 2019        |                                                              | <https://www.paperweekly.site/recommend-paper>               |
|      | Heterogeneous Graph Attention Networks for Semi-supervised<br/>Short Text Classification |          |       EMNLP 2019       |                                                              | s石川的工作，短文本分类因为semantic非常sparse，原来的工作又利用knowledge的side information，在这里作者提议利用短文本和entity以及topic之间的relation的关系，构成一幅异构图。相比于text gcn中 所有word和document都当作node，并且用one hot来表示，在这里，作者对每个node都使用了特征，这面临着他们特征不同，怎么进行卷积操作呢，在每层卷积中，它提议针对不同种类的节点之间的关系，以及不同的特征，把大图化成小图，并且不同的特征经过映射为同一特征空间，这和GCMC以及RGCN操作类似，另外，作者还有一个分层attention，一层是对连接种类做，然后基于此每一个节点会对其他所有节点进行attention。最后是softmax分类没什么说的 |
|      | Metapath-guided Heterogeneous Graph Neural Network for<br/>Intent Recommendation |          |        KDD 2019        |                                                              | z在itent 推荐场景下，user，item，query。是为了向用户推荐query。所有object都可以用word term来表示，可以大大减少embedding的参数数量，其余并没有太大改进。基本就是利用metapath得到用户和query的neighboor，然后利用GNN聚合得到两者的表示，然后做内积就是常见的作法。而metaph中，卷积只在query和item上来卷积，因为只有他们有共同的term来表示，而其他不共有的特征则是在最后将这些特征拼接上去。 |
|      | 2019 - KDD - NPA Neural News Recommendation with personalized attention |          |                        |                                                              | 作者的出发点是，同一篇文章被不同用户点击的原因不一样。user会有对应ID，映射为embedding，这个作为query vector会和文章中的每一个字作attention，这个title的表达就由这些字的受用户的attention来进行聚合。然后呢，用户的embedding还会再一次作为query vector和 不同的文章的表达来进行聚合，这些文章聚合后在作为用户的表达，和刚才的文章表达做内积即可。整个下来，需要存储的是user的embedding。整体下来文章是用户embedding和文章的内容交互后的vector来表达，而用户是有被用户embedding交互过的文章vector来进行表达。 |
|      | GNN Explainer                                                | Rex ying |       NIPS 2019        |                                                              | 以往的可解释方法主要用于解释特征，无法应用于GNN类网络，本文用mutual information去解释，GNN中，哪些点和哪些特征维度对当前的预测影响较大，有一个工具包LIME |













## Recomendation

|                            Title                             | **Conference** |                          **Author**                          |                        **Attachment**                        |
| :----------------------------------------------------------: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    **<font color=red>Large Scale Recommendation</font>**     |                |                                                              |                                                              |
| [Dynamic Attention Deep Model for Article Recommendation by Learning Human Editors’ Demonstration](https://dl.acm.org/citation.cfm?id=3098096) |    KDD 2017    |               Xuejian Wang, Lantao Yu, Kan Ren               |                                                              |
| [DKN: Deep Knowledge-Aware Network for News Recommendation ](https://dl.acm.org/citation.cfm?id=3186175) |    WWW 2018    |      [Hongwei Wang], Fuzheng Zhang, Xing Xie, Minyi Guo      |        [Tensorflow](https://github.com/hwwang55/DKN)         |
| [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) |    KDD 2018    |                 Guorui Zhou, Kun Gai, et al                  | [Tensorflow](https://github.com/zhougr1993/DeepInterestNetwork) |
|              **<font color=red>Normal</font>**               |                |                                                              |                                                              |
| [Learning Consumer and Producer Embeddings for User-Generated Content Recommendation](https://arxiv.org/abs/1809.09739) |  Recsys 2018   |             [Wang-Cheng Kang], [Julian McAuley]              |                                                              |
| [Spectral Collaborative Filtering](https://arxiv.org/abs/1808.10523v1) |  Recsys 2018   |              [Lei Zheng], Chun-Ta, Philip S. Yu              |                                                              |
| [Music Recommendation by Unified Hypergraph : Combining Social Media Information and Music Content](https://dl.acm.org/citation.cfm?id=1874005) |    MM 2010     |             Bu Jiajun, Tan Shulong, [Xiaofei He]             |                                                              |
|        **<font color=red>News Recommendation</font>**        |                |                                                              |                                                              |
| [News Recommendation via Hypergraph Learning: Encapsulation of User Behavior and News Content](https://dl.acm.org/citation.cfm?id=2433436) |   WSDM 2013    |                        Lei Li, Tao Li                        |                                                              |
| [Weave & Rec : A Word Embedding based 3-D Convolutional Network for News Recommendation]() |   CIKM 2018    |                                                              |        [Keras](https://github.com/dhruvkhattar/WE3CN)        |
|    **<font color=red>Review Based Recommendation</font>**    |                |                                                              |                                                              |
| [A3NCF: An Adaptive Aspect Attention Model for Rating Prediction](https://www.comp.nus.edu.sg/~xiangnan/papers/ijcai18-A3NCF.pdf) |   IJCAI 2018   | Zhiyong Cheng, Ying Ding, [Xiangnan He], Lei Zhu, Xuemeng Song, Mohan Kankanhalli |        [Keras](https://github.com/hustlingchen/A3NCF)        |
|                                                              |                |                                                              |                                                              |
| [Social Attentional Memory Network: Modeling Aspect- and Friend-level Differences in Recommendation](http://www.thuir.cn/group/~mzhang/publications/WSDM2019ChenChong.pdf) |   WSDM 2019    |                 Chong Chen, Min Zhang, et al                 |      [Tensorflow](https://github.com/chenchongthu/SAMN)      |
| [Graph Neural Networks for Social Recommendation](https://arxiv.org/pdf/1902.07243.pdf) |    WWW 2019    |             Wenqi Fan, [Yao Ma], [Jiliang Tang]              |                                                              |

### Explainable

- **Explainable Reasoning over Knowledge Graphs for Recommendation**
- **Explainable Recommendation Through Attentive Multi-View Learning (AAAI 2018)**
- **RippleNet : Propagating User Preferences on the Knowledge Graph for Recommender Systems (CIKM 2018)**

- [Min Zhang] website (aim at explainable recommender system)

## Graph

|                            Title                             |                        **Conference**                        |                          **Author**                          |                          Attachment                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|              **<font color=red>Survey</font>**               |                                                              |                                                              |                                                              |
| [Survey: Representation Learning on Graphs: Methods and Applications](https://arxiv.org/abs/1709.05584) |                                                              |      [William L. Hamilton], [Rex Ying], [Jure Leskovec]      |                                                              |
| [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596?context=cs) |                                                              |                   Zonghan Wu ,Philip S. Yu                   |                                                              |
|                                                              |                                                              |                                                              |                                                              |
|           **<font color=red>Graph Theory</font>**            |                                                              |                                                              |                                                              |
|                                                              |                                                              |                                                              |                                                              |
|                                                              |                                                              |                                                              |                                                              |
| [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536) |                          ICML 2018                           | [Keyulu Xu], Chengtao Li, Yonglong Tian, Tomohiro Sonobe,Ken-ichi Kawarabayashi, Stefanie Jegelka |                                                              |
| [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) |                          ICLR 2019                           | Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann  | [Pytorch and Tensorflow](https://github.com/klicperajo/ppnp) |
|                                                              |                                                              |                                                              |                                                              |
| [Bayesian graph convolutional neural networks for semi-supervised classification](https://arxiv.org/pdf/1811.11103.pdf) |                          AAAI 2019                           |        Jiatao Jiang, Zhen Cui, Chunyan Xu, Jian Yang         |                                                              |
| [Graph Wavelet Neural Network](https://openreview.net/forum?id=H1ewdiR5tQ) |                          ICLR 2019                           | [Bingbing Xu](https://openreview.net/profile?email=xubingbing%40ict.ac.cn)*,* [Huawei Shen](https://openreview.net/profile?email=shenhuawei%40ict.ac.cn)*,* [Qi Cao](https://openreview.net/profile?email=caoqi%40ict.ac.cn)*,* [Yunqi Qiu](https://openreview.net/profile?email=qiuyunqi%40ict.ac.cn)*,* [Xueqi Cheng](https://openreview.net/profile?email=cxq%40ict.ac.cn) | [Pytorch](https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork), [Tensorflow](https://github.com/Eilene/GWNN) |
| [GraphGAN: Graph Representation Learning with Generative Adversarial Nets](https://arxiv.org/pdf/1711.08267.pdf) |                          AAAI 2018                           | [Hongwei Wang], Jia Wang, Jialin Wang,Miao Zhao,Weinan Zhang,Fuzheng Zhang Xing Xie, Minyi Guo |      [Tensorflow](https://github.com/hwwang55/GraphGAN)      |
| [Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/pdf/1809.00130.pdf) |                          CIKM 2018                           |                     Ming Ding，Jie Tang                      |         [Code](https://github.com/dm-thu/GraphSGAN)          |
| [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) |                    ICML 2019 under review                    | Wu Felix, Zhang Tianyi, Souza, Amauri, Holanda de Fifty, Christopher, Yu, Tao, Weinberger, Kilian Q. |          [Pytorch](https://github.com/Tiiiger/SGC)           |
| [HOW POWERFUL ARE GRAPH NEURAL NETWORKS](https://openreview.net/forum?id=ryGs6iA5Km) |                          ICLR 2019                           |  [Keyulu Xu], Weihua Hu, [Jure Leskovec], Stefanie Jegelka   |                                                              |
| [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://github.com/naganandy/graph-based-deep-learning-literature/blob/master/conference-publications/folders/lanczosnet_iclr19/README.md) |                          ICLR 2019                           |                     [Renjie Liao], et al                     |      [code](https://github.com/lrjconan/LanczosNetwork)      |
| [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/abs/1802.00910)![1551274095870](ReadMe.assets/1551274095870.png) |                          AAAI 2019                           |                   Le Song, Yuan Qi, et al                    |                                                              |
| [Graph Attention Networks](https://arxiv.org/abs/1710.10903) |                          ICLR 2018                           | Petar Veliˇckovi´, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`, Yoshua Bengio |         [Tensorflow](https://github.com/PetarV-/GAT)         |
| [Hierarchical Graph Representation Learning with Differentiable Pooling](https://dl.acm.org/citation.cfm?id=2433436) |                          NIPS 2018                           | [Rex Ying], Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec |         [Code](https://github.com/RexYing/diffpool)          |
| [GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations](https://arxiv.org/abs/1806.05662) |                          NIPS 2018                           | Zhilin Yang, Jake Zhao, Bhuwan Dhingra, Kaiming He, William W. Cohen, Ruslan Salakhutdinov, Yann LeCun |                                                              |
| [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf) |                          NIPS 2017                           |                                                              |       [Code](http://snap.stanford.edu/graphsage/#code)       |
| [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868) |                          NIPS 2018                           |                   Shchur  Oleksandr et al                    | [Tensorflow & gnn bench mark](https://github.com/shchur/gnn-benchmark) |
| [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/abs/1609.02907) |                          ICLR 2017                           |               [Thomas N. Kipf],  [Max Welling]               |          [Tensorflow](https://github.com/tkipf/gcn)          |
|         **<font color=red>Graph Application</font>**         |                                                              |                                                              |                                                              |
| [DeepInf: Social Influence Prediction with Deep Learning](https://arxiv.org/pdf/1807.05560.pdf) |                           KDD 2018                           |              Jiezhong Qiu , [Jie Tang]， et al               |          [Pytorch](https://github.com/sunqm/pyscf)           |
| [Signed Graph Convolutional Network](https://arxiv.org/pdf/1808.06354.pdf) |                          ICDM 2018                           |               Tyler Derr, Yao Ma, Jiliang Tang               |    [Pytorch](https://github.com/benedekrozemberczki/SGCN)    |
| [Graph convolutional networks for text classification](https://arxiv.org/abs/1809.05679) |                          AAAI 2019                           |             Liang Yao, Chengsheng Mao, Yuan Luo              |     [Tensorflow](https://github.com/yao8839836/text_gcn)     |
| [Learning Graph Pooling and Hybrid Convolutional Operations for Text Representations](https://github.com/naganandy/graph-based-deep-learning-literature/blob/master/conference-publications/folders/gpool_www19/README.md) |                           WWW 2019                           |          Hongyang Gao,  Yongjun Chen,  Shuiwang Ji           |                                                              |
| [Graph Convolutional Matrix Completion](https://arxiv.org/pdf/1706.02263.pdf) |                           KDD 2018                           |    Rianne van den Berg, [Thomas N. Kipf],  [Max Welling]     |     [Tensorflow](https://github.com/riannevdberg/gc-mc)      |
| [PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://dl.acm.org/citation.cfm?id=3219890) |                           KDD 2018                           | [Rex Ying], Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, [Jure Leskovec] |                                                              |
|          **<font color=red>Knowledge Graph</font>**          |                                                              |                                                              |                                                              |
| [End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/abs/1811.04441) |                           AAAI2019                           |                                                              | [Pytorch](1https://github.com/JD-AI-Research-Silicon-Valley/SACN) |
| [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) |                          ESWC 2018                           | Michael Schlichtkrull, [Thomas N. Kipf], Peter Bloem, Rianne van den Berg, Ivan Titov, [Max Welling] | [Keras](https://github.com/tkipf/relational-gcn),[Tensorflow](https://github.com/MichSchli/RelationPrediction) |
| [SimplE Embedding for Link Prediction in Knowledge Graphs](https://arxiv.org/abs/1802.04868) |                          NIPS 2018                           |              [Seyed Mehran Kazemi], David Poole              |       [Tensorflow](https://github.com/Mehran-k/SimplE)       |
| [RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](https://arxiv.org/abs/1803.03467) |                          CIKM 2018                           | [Hongwei Wang], Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, [Xing Xie], Minyi Guo |     [Tensorflow](https://github.com/hwwang55/RippleNet)      |
| [DKN: Deep Knowledge-Aware Network for News Recommendation](https://dl.acm.org/citation.cfm?id=3186175) |                           WWW 2018                           |      [Hongwei Wang], Fuzheng Zhang, Xing Xie, Minyi Guo      |        [Tensorflow](https://github.com/hwwang55/DKN)         |
| [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476) |                          AAAI 2017                           | [Tim Dettmers], Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel |       [Pytorch](https://github.com/TimDettmers/ConvE)        |
| [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data) |                          NIPS 2013                           | Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko | [Code for TransE, TransH, TransR and PTransE](https://github.com/thunlp/KB2E) |
|                                                              |                                                              |                                                              |                                                              |
|            **<font color=red>HyperGraph</font>**             |                                                              |                                                              |                                                              |
| [Hypergraph Neural Networks](https://arxiv.org/abs/1809.09401) |                          AAAI 2019                           | Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, [Yue Gao] |         [Pytorch](https://github.com/Yue-Group/HGNN)         |
| [Structural Deep Embedding for Hyper-Networks](https://arxiv.org/abs/1711.10146) |                          AAAI 2018                           |      Ke Tu, [Peng Cui], Xiao Wang, Fei Wang, Wenwu Zhu       |        [Tensorflow](https://github.com/tadpole/DHNE)         |
| [Modeling Multi-way Relations with Hypergraph Embedding](https://dl.acm.org/citation.cfm?id=3269274) |                          CIKM 2018                           |   Chia-An Yu, Ching-Lun Tai, Tak-Shing Chan, Yi-Hsuan Yang   |           [matlab](http://github.com/chia-an/HGE)            |
| **<font color=red>Heterogeneous Information Network</font>** |                                                              |                                                              |                                                              |
| [Cash-out User Detection based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism](http://shichuan.org/doc/64.pdf) | Binbin Hu, Zhiqiang Zhang,[Chuan Shi], Jun Zhou, Xiaolong Li, Yuan Qi |                          AAAI 2019                           | [Code will be released ?](https://github.com/librahu/HACUD)  |
| [Heterogeneous Graph Attention Network](http://pengcui.thumedialab.com/papers/HeterogeneousGAN.pdf) |                           WWW 2019                           |                [Chuan Shi], [Peng Cui] et al                 |          [Pyotrch](https://github.com/Jhy1993/HAN)           |
| [Relation Structure-Aware Heterogeneous Information Network Embedding](http://shichuan.org/doc/63.pdf) |                          AAAI 2019                           |        Yuanfu Lu, [Chuan Shi], Linmei Hu, Zhiyuan Liu        |          [Pytorch](https://github.com/rootlu/RHINE)          |
| [Are Meta-Paths Necessary ? Revisiting Heterogeneous Graph Embeddings](https://dl.acm.org/citation.cfm?id=3271777) |                          CIKM 2018                           |                        [Rana Hussein]                        |                       Request in email                       |
| [Leveraging Meta-path based Context for Top- N Recommendation with A Neural Co-Attention Model](http://shichuan.org/doc/47.pdf) |                           KDD 2018                           |    Binbin Hu, [Chuan Shi], Wayne Xin Zhao, [Philip S. Yu]    | [Tensorflow&Keras](https://github.com/librahu/MCRec),[Data](https://github.com/librahu/Heterogeneous-Information-Network-Datasets-for-Recommendation) |
| [PME : Projected Metric Embedding on Heterogeneous Networks for Link Prediction](https://dl.acm.org/citation.cfm?id=3219986) |                           KDD 2018                           | [ Hongxu Chen](https://dl.acm.org/author_page.cfm?id=99659155111&coll=DL&dl=ACM&trk=0) et al |                       Request in email                       |
| [Aspect-Level Deep Collaborative Filtering via Heterogeneous Information Networks](http://shichuan.org/doc/46.pdf) |                          IJCAI 2018                          | Xiaotian Han, [Chuan Shi], Senzhang Wang, [Philip S. Yu], Li Song |         [Tensorflow](https://github.com/ahxt/NeuACF)         |
| [Deep Collective Classification in Heterogeneous Information Networks](https://dl.acm.org/authorize?N655839) |                           WWW 2018                           |                                                              | [Keras](https://github.com/zyz282994112/GraphInception.git)  |
| [Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks](http://www.cse.ust.hk/~hzhaoaf/data/kdd17-paper.pdf) |                           KDD 2017                           | [Huan Zhao], anming Yao, Jianda Li, Yangqiu Song and Dik Lun Lee |       [Python](https://github.com/HKUST-KnowComp/FMG)        |
| [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) |                           KDD 2017                           |                        [Yuxiao Dong]                         | [C++](https://ericdongyx.github.io/metapath2vec/m2v.html), [Tensorflow](https://github.com/apple2373/metapath2vec) |
|       **<font color=red>Hyperbolic embedding</font>**        |                                                              |                                                              |                                                              |
| [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) |                          NIPS 2017                           |               [Maximilian Nickel], Kiela Douwe               | [Pytorch](https://github.com/facebookresearch/poincare-embeddings) |
| [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112) |                          NIPS 2018                           |           [Octavian Eugen Ganea], Hofmann, Thomas            |     [Tensorflow](https://github.com/dalab/hyperbolic_nn)     |
| **<font color=red>Network Representation Learning (unsupervised)</font>** |                                                              |                                                              |                                                              |
| [A Survey on Network Embedding](https://arxiv.org/abs/1711.08752) |                             2017                             | [Peng Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+P), [Xiao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Jian Pei](https://arxiv.org/search/cs?searchtype=author&query=Pei%2C+J), [Wenwu Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W) |                                                              |
| [A Tutorial on Network Embeddings](https://arxiv.org/abs/1808.02590) |                             2018                             | [Haochen Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+H), [Bryan Perozzi](https://arxiv.org/search/cs?searchtype=author&query=Perozzi%2C+B), [Rami Al-Rfou](https://arxiv.org/search/cs?searchtype=author&query=Al-Rfou%2C+R), [Steven Skiena](https://arxiv.org/search/cs?searchtype=author&query=Skiena%2C+S) |                                                              |
| [TransNet : Translation-Based Network Representation Learning for Social Relation Extraction](https://www.ijcai.org/proceedings/2017/0399.pdf) |                          IJCAI 2017                          |              Cunchao Tu, Zhengyan, Maosong Sun               |       [Tensorflow](https://github.com/thunlp/TransNet)       |
| [TransConv: Relationship Embedding in Social Networks](https://www.aaai.org/Papers/AAAI/2019/AAAI-LaiYi-Yu.6773.pdf) |                        AAAI 2019 oral                        |                                                              |                                                              |
|  [DEEP GRAPH INFOMAX](https://arxiv.org/pdf/1809.10341.pdf)  |                          ICLR 2019                           | Petar Velickovi ˇ c´, William L. Hamilton, [Yoshua Bengio] et al |          [Pytorch](https://github.com/PetarV-/DGI)           |
| [ANRL: Attributed Network Representation Learning via Deep Neural Networks](https://www.ijcai.org/proceedings/2018/0438.pdf) |                          IJCAI 2018                          | Zhen Zhang, Hongxia Yang, Jiajun Bu, Sheng Zhou, Pinggang Yu, Jianwei Zhang, Martin Ester, Can Wang |      [Tensosrflow](https://github.com/cszhangzhen/ANRL)      |
- **Exploiting Relational Information in Social Networks using Geometric Deep Learning on Hypergraphs**


  **Other implement resource**:

  - [gated-graph-neural-network-samples](https://github.com/Microsoft/gated-graph-neural-network-samples)
  - [Graph-neural-networks jupyter tutorial](https://github.com/SeongokRyu/Graph-neural-networks)
  - [Deep Graph Library (DGL) Python package](https://docs.dgl.ai/index.html)
  - [pitafall: gnn model collection](https://github.com/shchur/gnn-benchmark)
  - [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
  - [alimama euler framework](https://github.com/alibaba/euler)
  - [tencent angel frame work](<https://github.com/Angel-ML/angel>)
  - [Liaojunjie: gnn model collection](https://github.com/lrjconan/LanczosNetwork)
  - [node embedding from deepwalk to struc2vec](https://github.com/shenweichen/GraphEmbedding)
  - [spektral](https://github.com/danielegrattarola/spektral)

  **Other reading materials**:

  - [Tsinghua University Graph papers reading list](https://github.com/thunlp/GNNPapers)
  - [gnn literature](https://github.com/naganandy/graph-based-deep-learning-literature/blob/master/conference-publications/README.md)
  - [MIA reading group](https://github.com/shagunsodhani/Graph-Reading-Group)
  - [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
  - [dynamic graph](https://github.com/woojeongjin/dynamic-KG)
  - [zhihu link for graph](https://zhuanlan.zhihu.com/p/55944583)
  - [spatial-temporal graph](https://github.com/Eilene/spatio-temporal-paper-list/issues/1)
  - [Technische Universität München](https://www.kdd.in.tum.de/forschung/machine-learning-for-graphsnetworks/)

## BayesianDeepLearning

| Title                                                        | Conference | Author                                         | Attachment |
| ------------------------------------------------------------ | ---------- | ---------------------------------------------- | ---------- |
| **<font color=red>Survey</font>**                            |            |                                                |            |
| [Recent Advances in Autoencoder-Based Representation Learning](https://arxiv.org/abs/1812.05069) | NIPS 2018  | Michael Tschannen, Olivier Bachem, Mario Lucic |            |

## Datasets

### homegenerous graph dataset
- **PubMed Diabetes**
  - The Pubmed Diabetes dataset consists of 19717 scientific publications from PubMed database pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words. The README file in the dataset provides more details.
  - Download Link:
    - https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz
  - Related Papers:
    - Galileo Namata, et. al. "Query-driven Active Surveying for Collective Classification." MLG. 2012.

- **Cora**
  - The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. The README file in the dataset provides more details.
  - Download Link:
    - https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
  - Related Papers:
    - Qing Lu, and Lise Getoor. "Link-based classification." ICML, 2003.
    - Prithviraj Sen, et al. "Collective classification in network data." AI Magazine, 2008.

other useful datasets link:
- citation dataset
  - https://linqs.soe.ucsc.edu/data


### heteregeneous graph datasets

- **IMDB Datasets**
  - MovieLens Latest Dataset which consists of 33,000 movies. And it contains four types of nodes: movie, director, actor and actress, connected by two types of relations/link: directed link and actor/actress staring link. Each movie is assigned with a set of class labels, indicating generes of the movie. For each movie, we extract a bag of words vector of all the plot summary about the movie as local features, which include 1000 words.
  - Download Link:
    - https://github.com/trangptm/Column_networks/tree/master/data
  - Related Papers:
    - T. Pham, et al. "Column networks for collective classification." In AAAI, 2017.
    - Zhang, Yizhou et al. "Deep Collective Classification in Heterogeneous Information Networks" In WWW, 2018.


other useful dataset links

- processed Datasets
  - https://github.com/librahu/Heterogeneous-Information-Network-Datasets-for-Recommendation-and-Network-Embedding/blob/master/README.md

[Octavian Eugen Ganea]: http://people.inf.ethz.ch/ganeao/
[Maximilian Nickel]: https://mnick.github.io/project/geometric-representation-learning/

[Chuan Shi]: http://shichuan.org/ShiChuan_ch.html
[Xing Xie]: https://www.microsoft.com/en-us/research/people/xingx/#!representative-publications
[Tim Dettmers]: http://timdettmers.com/about/
[Seyed Mehran Kazemi]: https://www.cs.ubc.ca/~smkazemi/
[Xiangnan He]: https://www.comp.nus.edu.sg/~xiangnan/
[Hongwei Wang]: https://hwwang55.github.io/
[Peng Cui]: http://pengcui.thumedialab.com/
[William L. Hamilton]: https://williamleif.github.io/
[Yue Gao]: http://www.gaoyue.org/tsinghua/pubs/index.htm
[Xiaofei He]: http://www.cad.zju.edu.cn/home/xiaofeihe/
[Thomas N. Kipf]: https://tkipf.github.io/
[Max Welling]: https://staff.fnwi.uva.nl/m.welling/
[Rex Ying]: https://cs.stanford.edu/people/rexy/
[Lei Zheng]: https://lzheng21.github.io/publications/
[Jure Leskovec]: https://cs.stanford.edu/~jure/
[Wang-Cheng Kang]: http://cseweb.ucsd.edu/~wckang/
[Julian McAuley]:  https://cseweb.ucsd.edu/~jmcauley/
[Rana Hussein]:  https://exascale.info/members/rana-hussein/
[Yuxiao Dong]:  https://ericdongyx.github.io/
[Renjie Liao]: http://www.cs.toronto.edu/~rjliao/
[Min Zhang]: http://www.thuir.org/group/~mzhang/
[Yao Ma]: http://cse.msu.edu/~mayao4/publications.html
[Jiliang Tang]: http://www.cse.msu.edu/~tangjili/publication.html
[jian tang]:  https://jian-tang.com/
[Keyulu Xu]: http://keyulux.com/



