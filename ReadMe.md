# Awesome paper list

[![PRs Welcome](ReadMe.assets/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

<p align="center">
  <img width="460" src="Word Art.png">
</p>

A collection of graph embedding, deep learning, recommendation, knowledge graph, heterogeneous graph papers with reference implementations



Table of Contents
=================

   * [Awesome paper list](#awesome-paper-list)
   * [Table of Contents](#table-of-contents)
      * [1 Recommendation](#1-recommendation)
         * [1.1 Large Scale Recommendation](#11-large-scale-recommendation)
         * [1.2 Novel Application](#12-novel-application)
         * [1.3 News Recommendation](#13-news-recommendation)
         * [1.3 Social Recommendation](#13-social-recommendation)
         * [1.4 Disentangled Recommendation](#14-disentangled-recommendation)
         * [1.5 Explainable Recommendation](#15-explainable-recommendation)
      * [2 Graph](#2-graph)
         * [2.1 Survey](#21-survey)
         * [2.2 Theory](#22-theory)
         * [2.3 Application](#23-application)
            * [2.3.1 Knowledge Graph](#231-knowledge-graph)
         * [2.4 HyperGraph](#24-hypergraph)
         * [2.5 Heterogeneous Information Network](#25-heterogeneous-information-network)
            * [2.5.1 Architecture](#251-architecture)
            * [2.5.2 Recommendation and Other Application](#252-recommendation-and-other-application)
         * [2.6 Network Representation Learning](#26-network-representation-learning)
            * [2.6.1 Survey](#261-survey)
         * [2.7 Sources](#27-sources)
            * [2.7.1 Industry Implement](#271-industry-implement)
            * [2.7.2 Acdamic Implement](#272-acdamic-implement)
            * [2.7.3 Reading Source](#273-reading-source)
         * [2.8 Graph Modification and Robust](#28-graph-modification-and-robust)
         * [2.9 Understanding](#29-understanding)
         * [2.10 Sampling](#210-sampling)
      * [3 BayesianDeepLearning](#3-bayesiandeeplearning)
         * [3.1 Survey](#31-survey)
         * [3.2 Uncertainty](#32-uncertainty)
      * [4 Others](#4-others)
      * [5 Datasets](#5-datasets)
         * [5.1 homegenerous graph dataset](#51-homegenerous-graph-dataset)
         * [5.2 heteregeneous graph datasets](#52-heteregeneous-graph-datasets)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)



## 1 Recommendation

### 1.1 Large Scale Recommendation

- 2017- KDD -  Dynamic Attention Deep Model for Article Recommendation by Learning Human Editors’ Demonstration
  - *Xuejian Wang, Lantao Yu, Kan Ren*
  - *news recommendation*

- 2018 - WWW - [DKN: Deep Knowledge-Aware Network for News Recommendation](https://github.com/hwwang55/DKN)
  - *[Hongwei Wang], Fuzheng Zhang, Xing Xie, Minyi Guo*
  - *news recommendation; knowledge graph*


- 2018 - KDD - [Deep Interest Network for Click-Through Rate Prediction](https://github.com/zhougr1993/DeepInterestNetwork)
  - *Guorui Zhou, Kun Gai, et al*
  - *click prediction*

### 1.2 Novel Application

- 2018 - Recsys - Learning Consumer and Producer Embeddings for User-Generated Content Recommendation
  - *[Wang-Cheng Kang], [Julian McAuley]*
  - *user based*

- 2019 - ICML - Compositional Fairness Constraints for Graph Embeddings

### 1.3 News Recommendation

- 2019 - KDD - NPA Neural News Recommendation with personalized attention

- 2013 -WSDM - News Recommendation via Hypergraph Learning: Encapsulation of User Behavior and News Content
  - *Lei Li, Tao Li*

- 2018 - CIKM - [Weave & Rec : A Word Embedding based 3-D Convolutional Network for News Recommendation](https://github.com/dhruvkhattar/WE3CN)

- 2018 - IJCAI - [A3NCF: An Adaptive Aspect Attention Model for Rating Prediction](https://github.com/hustlingchen/A3NCF)
  - *Zhiyong Cheng, Ying Ding, [Xiangnan He], Lei Zhu, Xuemeng Song*

### 1.3 Social Recommendation

- 2019 - WSDM - [Social Attentional Memory Network: Modeling Aspect- and Friend-level Differences in Recommendation](https://github.com/chenchongthu/SAMN)
  - *Chong Chen, Min Zhang, et al*

- 2019 - WWW - Graph Neural Networks for Social Recommendation
  - *Wenqi Fan, [Yao Ma], [Jiliang Tang]*

- 2019 - NIPS - [Disentangled Graph Convolutional Networks](https://jianxinma.github.io/)
  - *[Jianxin Ma, Peng Cui]*

### 1.4 Disentangled Recommendation

- 2019 - NIPS - [Disentangled Graph Convolutional Networks](https://jianxinma.github.io/)
  - *[Jianxin Ma, Peng Cui]*

### 1.5 Explainable Recommendation

- 2018 - AAAI - Explainable Recommendation Through Attentive Multi-View Learning

- 2018 - CIKM - RippleNet : Propagating User Preferences on the Knowledge Graph for Recommender Systems

- 2019 - AAAI - Explainable Reasoning over Knowledge Graphs for Recommendation

- [Min Zhang] website (aim at explainable recommender system)

## 2 Graph

### 2.1 Survey
- 2019  - Representation Learning on Graphs: Methods and Applications
  - *[William L. Hamilton], [Rex Ying], [Jure Leskovec]*

- 2019  - A Comprehensive Survey on Graph Neural Networks
  - *Zonghan Wu ,Philip S. Yu*

### 2.2 Theory



- 2019 - ICML - [Disentangled Graph Convolutional Networks](https://jianxinma.github.io/)
  - *Jinxi Ma, [Peng Cui]*

- 2018 - ICML - Representation Learning on Graphs with Jumping Knowledge Networks
  - *[Keyulu Xu], Chengtao Li, Yonglong Tian, Tomohiro Sonobe,Ken-ichi Kawarabayashi, Stefanie Jegelka*
  - *jump connection;*
  
- 2019 - ICLR - [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://github.com/klicperajo/ppnp)
  - *Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann*
  - *page rank;*

- 2019 - ICLR - [Graph Wavelet Neural Network](https://github.com/Eilene/GWNN)
  
  - *[Bingbing Xu](https://openreview.net/profile?email=xubingbing%40ict.ac.cn), [Huawei Shen](https://openreview.net/profile?email=shenhuawei%40ict.ac.cn), [Qi Cao](https://openreview.net/profile?email=caoqi%40ict.ac.cn), [Yunqi Qiu](https://openreview.net/profile?email=qiuyunqi%40ict.ac.cn), [Xueqi Cheng](https://openreview.net/profile?email=cxq%40ict.ac.cn)*
  
- 2018 - AAAI - [GraphGAN: Graph Representation Learning with Generative Adversarial Nets](https://github.com/hwwang55/GraphGAN)
  - *[Hongwei Wang], Jia Wang, Jialin Wang,Miao Zhao,Weinan Zhang,Fuzheng Zhang Xing Xie, Minyi Guo*
  
- 2018 - CIKM - [Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://github.com/dm-thu/GraphSGAN)  
  - *[Hongwei Wang], Jia Wang, Jialin Wang,Miao Zhao,Weinan Zhang,Fuzheng Zhang Xing Xie, Minyi Guo*
  
- 2019 - ICML - [Simplifying Graph Convolutional Networks](https://github.com/Tiiiger/SGC)
  - *Wu Felix, Zhang Tianyi, Souza, Amauri, Holanda de Fifty, Christopher, Yu, Tao, Weinberger, Kilian Q.*
  
- 2019 - ICLR - [HOW POWERFUL ARE GRAPH NEURAL NETWORKS](https://github.com/Tiiiger/SGC)
  - *[Keyulu Xu], Weihua Hu, [Jure Leskovec], Stefanie Jegelka*
  
- 2019 - ICLR - [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](https://github.com/lrjconan/LanczosNetwork)
  - *[Renjie Liao], et al*
  
- 2019 - AAAI - [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://github.com/lrjconan/LanczosNetwork)
  - *Le Song, Yuan Qi, et al*
  
- 2018 - ICLR - [Graph Attention Networks](https://github.com/PetarV-/GAT)
  - *Petar Veliˇckovi´, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`, Yoshua Bengio*
  
- 2018 - NIPS - [Hierarchical Graph Representation Learning with Differentiable Pooling](https://github.com/RexYing/diffpool)
  - *[Rex Ying], Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec*
  
- 2018 - NIPS - GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations
  - *Zhilin Yang, Jake Zhao, Bhuwan Dhingra, Kaiming He, William W. Cohen, Ruslan Salakhutdinov, Yann LeCun*
  
- 2017 - NIPS - [GraphSAGE: Inductive Representation Learning on Large Graphs](http://snap.stanford.edu/graphsage/#code)
  - *[Rex Ying], Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec*
  
- 2018 - NIPS - [Pitfalls of Graph Neural Network Evaluation](https://github.com/shchur/gnn-benchmark)
  - *Shchur  Oleksandr et al*

- 2017 - ICLR - [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://github.com/tkipf/gcn)
  - *[Thomas N. Kipf],  [Max Welling]*

### 2.3 Application

- 2018 - KDD - [DeepInf: Social Influence Prediction with Deep Learning](https://github.com/sunqm/pyscf)
  - *Jiezhong Qiu , Jie Tang， et al*

- 2018 - KDD - [Signed Graph Convolutional Network](https://github.com/benedekrozemberczki/SGCN)
  - *yler Derr, Yao Ma, Jiliang Tang*

- 2019 - AAAI - [Signed Graph Convolutional Network](https://github.com/yao8839836/text_gcn)
  - *Liang Yao, Chengsheng Mao, Yuan Luo*

- 2018 - KDD - [Graph Convolutional Matrix Completion](https://github.com/riannevdberg/gc-mc)
  - *Rianne van den Berg, [Thomas N. Kipf],  [Max Welling]*

- 2018 - KDD - PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems
  - *[Rex Ying], Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, [Jure Leskovec]*

#### 2.3.1 Knowledge Graph

- 2019 - AAAI - [End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://github.com/JD-AI-Research-Silicon-Valley/SACN)

- 2019 - AAAI - [Modeling Relational Data with Graph Convolutional Networks](https://github.com/tkipf/relational-gcn)
  - *Michael Schlichtkrull, [Thomas N. Kipf]*

- 2018 - NIPS - [SimplE Embedding for Link Prediction in Knowledge Graphs](https://github.com/Mehran-k/SimplE)
  - *[Seyed Mehran Kazemi], David Poole*

- 2017 - AAAI - [Convolutional 2D Knowledge Graph Embeddings](https://github.com/TimDettmers/ConvE)
  - *[Tim Dettmers], Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel*

- 2013 - NIPS - [Translating Embeddings for Modeling Multi-relational Data](https://github.com/thunlp/KB2E)
  - *Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko*

### 2.4 HyperGraph

- 2019 - AAAI - [Hypergraph Neural Networks](https://github.com/Yue-Group/HGNN)
  - *Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, [Yue Gao]*

- 2018 - AAAI - [Structural Deep Embedding for Hyper-Networks](https://github.com/tadpole/DHNE)
  - *Ke Tu, [Peng Cui], Xiao Wang, Fei Wang, Wenwu Zhu*


### 2.5 Heterogeneous Information Network

#### 2.5.1 Architecture

- 2019 - NIPS - [Graph Transformer Networks](https://github.com/seongjunyun/Graph_Transformer_Networks)
  - *Ke Tu, [Peng Cui], Xiao Wang, Fei Wang, Wenwu Zhu*

- 2019 - WWW - [Heterogeneous Graph Attention Network](https://github.com/Jhy1993/HAN)
  - *Houye Ji, [Chuan Shi], [Peng Cui] et al*

- 2019 - AAAI - [Relation Structure-Aware Heterogeneous Information Network Embedding](https://github.com/rootlu/RHINE)
  - *Yuanfu Lu, [Chuan Shi], Linmei Hu, Zhiyuan Liu*

- 2018 - CIKM - Are Meta-Paths Necessary ? Revisiting Heterogeneous Graph Embeddings
  - *[Rana Hussein]*

- 2018 - WWW - [Deep Collective Classification in Heterogeneous Information Networks](https://github.com/zyz282994112/GraphInception.git)
  - *[Rana Hussein]*

- 2018 - KDD - PME : Projected Metric Embedding on Heterogeneous Networks for Link Prediction
  - *[Hongxu Chen]*

- 2017 - KDD - [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://github.com/apple2373/metapath2vec)
  - *[Yuxiao Dong]* 	

#### 2.5.2 Recommendation and Other Application

- 2019 - CIKM - [Relation-Aware Graph Convolutional Networks for Agent-Initiated Social E-Commerce Recommendation](https://github.com/xfl15/RecoGCN)
  - *neighborhood sampling;*

- 2019 - KDD - Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation

- 2019 -EMNLP - Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification


- 2017 - KDD - [Meta-Graph Based Recommendation Fusion over Heterogeneous Information Networks](https://github.com/HKUST-KnowComp/FMG)
  - *Huan Zhao, anming Yao, Jianda Li, Yangqiu Song and Dik Lun Lee*

- 2019 - AAAI - [Cash-out User Detection based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism](https://github.com/librahu/HACUD)
  - *Binbin Hu, Zhiqiang Zhang, [Chuan Shi], Jun Zhou, Xiaolong Li, Yuan Qi*

- 2018 - KDD - [Leveraging Meta-path based Context for Top- N Recommendation with A Neural Co-Attention Model](https://github.com/librahu/MCRec)
  - *Binbin Hu, [Chuan Shi], Wayne Xin Zhao, [Philip S. Yu]*

- 2018 - IJCAI - [Aspect-Level Deep Collaborative Filtering via Heterogeneous Information Networks](https://github.com/ahxt/NeuACF)
  - *Xiaotian Han, [Chuan Shi], Senzhang Wang, [Philip S. Yu], Li Song*


### 2.6 Network Representation Learning

#### 2.6.1 Survey

- 2018 - A Survey on Network Embedding
  - *[Peng Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+P), [Xiao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Jian Pei](https://arxiv.org/search/cs?searchtype=author&query=Pei%2C+J), [Wenwu Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W)*

- 2018 - A Survey on Network Embedding
  - *[Peng Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+P), [Xiao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Jian Pei](https://arxiv.org/search/cs?searchtype=author&query=Pei%2C+J), [Wenwu Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+W)*

- 2018 - A Tutorial on Network Embeddings
  - *Haochen Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+H), [Bryan Perozzi](https://arxiv.org/search/cs?searchtype=author&query=Perozzi%2C+B), [Rami Al-Rfou](https://arxiv.org/search/cs?searchtype=author&query=Al-Rfou%2C+R), [Steven Skiena](https://arxiv.org/search/cs?searchtype=author&query=Skiena%2C+S)*

- 2017 - IJCAI - [TransNet : Translation-Based Network Representation Learning for Social Relation Extraction](https://github.com/thunlp/TransNet)
  - *Cunchao Tu, Zhengyan, Maosong Sun* 

- 2019 - AAAI - TransConv: Relationship Embedding in Social Networks

- 2019 - ICLR - [DEEP GRAPH INFOMAX](https://github.com/PetarV-/DGI)
  - Petar Velickovi ˇ c´, William L. Hamilton, [Yoshua Bengio] et al

- 2018 IJCAI - [ANRL: Attributed Network Representation Learning via Deep Neural Networks](https://github.com/cszhangzhen/ANRL)
  - *Zhen Zhang, Hongxia Yang, Jiajun Bu, Sheng Zhou, Pinggang Yu, Jianwei Zhang, Martin Ester, Can Wang*

### 2.7 Sources

#### 2.7.1 Industry Implement

  - [alimama euler framework](https://github.com/alibaba/euler)
  - [tencent angel frame work](<https://github.com/Angel-ML/angel>)
  - [tencent plato]( https://github.com/Tencent/plato )

#### 2.7.2 Acdamic Implement
  - [gated-graph-neural-network-samples](https://github.com/Microsoft/gated-graph-neural-network-samples)
  - [Graph-neural-networks jupyter tutorial](https://github.com/SeongokRyu/Graph-neural-networks)
  - [Deep Graph Library (DGL) Python package](https://docs.dgl.ai/index.html)
  - [pitafall: gnn model collection](https://github.com/shchur/gnn-benchmark)
  - [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
  - [Liaojunjie: gnn model collection](https://github.com/lrjconan/LanczosNetwork)
  - [node embedding from deepwalk to struc2vec](https://github.com/shenweichen/GraphEmbedding)
  - [spektral](https://github.com/danielegrattarola/spektral)
  - **[stellargraph including metapath2vec](https://github.com/stellargraph/stellargraph)**
  - [visualization of graph- graph tool]( https://graph-tool.skewed.de/ )
  - [analysis the spectral of graph pyqsp]( https://github.com/epfl-lts2/pygsp )

#### 2.7.3 Reading Source
  - [Tsinghua University Graph papers reading list](https://github.com/thunlp/GNNPapers)
  - [gnn literature](https://github.com/naganandy/graph-based-deep-learning-literature/blob/master/conference-publications/README.md)
  - [MIA reading group](https://github.com/shagunsodhani/Graph-Reading-Group)
  - [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
  - [dynamic graph](https://github.com/woojeongjin/dynamic-KG)
  - [zhihu link for graph](https://zhuanlan.zhihu.com/p/55944583)
  - [spatial-temporal graph](https://github.com/Eilene/spatio-temporal-paper-list/issues/1)
  - [Technische Universität München](https://www.kdd.in.tum.de/forschung/machine-learning-for-graphsnetworks/)

### 2.8 Graph Modification and Robust 

- 2019 - NIPS - [Graph Agreement Models for Semi-Supervised Learning](https://github.com/tensorflow/neural-structured-learning/tree/master/research/gam)
  - *Otilia Stretcu · Krishnamurthy Viswanathan · Dana Movshovitz-Attias · Emmanouil Platanios · Sujith Ravi · Andrew*
  - *self learning; edge  modification*

- 2019 - AAAI - [Bayesian graph convolutional neural networks for semi-supervised classification](https://github.com/huawei-noah/BGCN)
  - *Jiatao Jiang, Zhen Cui, Chunyan Xu, Jian Yang*
  - *edge modification;*

- 2019 - ICML - Learning Discrete Structures for Graph Neural Networks

- 2019 - ICML - Are Graph Neural Networks Miscalibrated

- 2019 - ICLR - DEEP GAUSSIAN EMBEDDING OF GRAPHS UNSUPERVISED INDUCTIVE LEARNING VIA RANKING
  - *提出概率建模，variance对应有具体的含义，另外提出ranking，即1 hop node embedding的similarity要大于2 hop node embedding，利用KL来计算similarity。同时，相比于node2vec这样的node embedding算法，该算法能够利用node attribute做到inductive，相比于graph sage，能够做到在test阶段，即使没有link，也能够产生node 的 embedding*
- 2019 - KDD - Robust Graph Convolutional Networks Against Adversaria Attacks  
  - *gcn中每一层特征都用一个gaussian distribution来表征，分布的好处是可以吸收对抗攻击的坏处。另外，设计了基于variance的attention方式，这样可以阻止对抗攻击在graph中的传播*

### 2.9 Understanding 

- 2020 - ICLR - Characterize and Transfer Attention in Graph Neural Networks 
  - *GAT在citation数据集上不同node的attention区分度不明显，在PPI上明显。这个attention和不同的数据集有着相关性，利用attention score作为feature vector，可以明显的区分出来不同的dataset。另外，作者尝试利用GAT得到的attention score对edge进行过滤，发现graph中的仅仅保留30-40%边仍能够得到不错的效果*

- 2020 - AAAI - Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View
  - *作者发现对于node classification这样的任务，inter class edge是有用的，而intra classi的edge是noisy的。作者提供了两种指标来衡量smoothing。同时作者还提出了两种方法来解决oversmooting，一种是加regularizer，在graph较近的node之间的feature vector的cosine distance变小，而graph上离得比较远的node之间的distance变大，另外一种方法为对graph进行重建，期望graph之间confidence比较高的edge得以保留，confidence比较低的边去除掉。这两种方法来使得在达到较高的层数的时候，performance的衰退变慢。*

### 2.10 Sampling
- 2019 - KDD - [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://www.paperweekly.site/papers/3251)

## 3 BayesianDeepLearning

### 3.1 Survey

- 2018 - NIPS - Recent Advances in Autoencoder-Based Representation Learning
  - *Michael Tschannen, Olivier Bachem, Mario Lucic*

- 2017 - ICML - [On Calibration of Modern Neural Networks](https://github.com/gpleiss/temperature_scaling)

- 2019 - NIPS - Variational Graph Convolutional Networks

### 3.2 Uncertainty

- 2017 - NIPS - What Uncertainties Do We Need in Bayesian  Deep Learning for Computer Vision?

- 2016 - ICML - Dropout as a Bayesian Approximation Representing Model Uncertainty in Deep Learning

- 2019 - NIPS - [Uncertainty posters](https://nips.cc/Conferences/2019/ScheduleMultitrack?session=15553)

- 2019 - ICLR - [Modeling Uncertainty with Hedged Instance Embedding](https://github.com/google/n-digit-mnist)

- 2019 - thisis - Uncertainty Quantification in Deep Learning

- 2019 - [Uncertainty Quantification in Deep Learning(https://www.inovex.de/blog/uncertainty-quantification-deep-learning/)

2019 - NIPS - Practical Deep Learning with Bayesian Principles

## 4 Others

2020 - CVPR - [CONTRASTIVE REPRESENTATION DISTILLATION](https://github.com/HobbitLong/RepDistiller)   

2017 - NIPS - Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

2019 - NIPS - A Simple Baseline for Bayesian Uncertainty in Deep Learning

2020 - AISTATS - [Confident Learning Estimating Uncertainty in Dataset Labels] (https://github.com/cgnorthcutt/cleanlab)

## 5 Datasets

### 5.1 homegenerous graph dataset

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

### 5.2 heteregeneous graph datasets

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
[Philip S. Yu]:  https://www.cs.uic.edu/~psyu

