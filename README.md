# Deep Reinforcement Learning for Recommender Systems 

This Work Collected the Courses / Books on Deep Reinforcement Learning (DRL) and DRL papers for recommender system.  

**(This work is forked from https://github.com/cszhangzhen/DRL4Recsys. In this fork, we updated some latest paper and added some comments or summaries)**

## Courses
#### UCL Course on RL 
[http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
#### CS 294-112 at UC Berkeley
[http://rail.eecs.berkeley.edu/deeprlcourse/](http://rail.eecs.berkeley.edu/deeprlcourse/)
#### Stanford CS234: Reinforcement Learning
[http://web.stanford.edu/class/cs234/index.html](http://web.stanford.edu/class/cs234/index.html)

## Book
1. **Reinforcement Learning: An Introduction (Second Edition)**. Richard S. Sutton and Andrew G. Barto. [book](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

1. **强化学习实战：强化学习在阿里的技术演进和业务创新** ISBN：9787121338984

## Papers
### Survey Papers
1. **A Brief Survey of Deep Reinforcement Learning**. Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage, Anil Anthony Bharath. 2017. [paper](https://arxiv.org/pdf/1708.05866.pdf)

1. **Deep Reinforcement Learing: An Overview**. Yuxi Li. 2017. [paper](https://arxiv.org/pdf/1701.07274.pdf)

1. **Deep Reinforcement Learning for Search, Recommendation, and Online Advertising: A Survey**. . Sigweb 19. [paper](https://arxiv.org/pdf/1812.07127.pdf)

### Conference Papers
#### <2018
1. **An MDP-Based Recommender System**. Guy Shani, David Heckerman, Ronen I. Brafman. JMLR 2005. [paper](http://www.jmlr.org/papers/volume6/shani05a/shani05a.pdf)  
*MDP*

1. **Usage-Based Web Recommendations: A Reinforcement Learning Approach**. Nima Taghipour, Ahmad Kardan, Saeed Shiry Ghidary. Recsys 2007. [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.9640&rep=rep1&type=pdf)

1. **A hybrid web recommender system based on q-learning**. Nima Taghipour and Ahmad Kardan. 2008. In Proceedings of the 2008 ACM symposium on Applied computing. ACM, 1164–1168. [paper](http://home.ustc.edu.cn/~xhlm630/RL_Research/A_Hybrid_Web_Recommender_System_Based_on_Q-Learning.pdf)  
*Q-learning*

1. **DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation**. Elad Liebman, Maytal Saar-Tsechansky, Peter Stone. AAMAS 2015. [paper](https://arxiv.org/pdf/1401.1880.pdf)

1. ★★★  
**Deep Reinforcement Learning for List-wise Recommendations**. Xiangyu Zhao, Liang Zhang, Zhuoye Ding, Dawei Yin, Yihong Zhao, and Jiliang Tang. 2017.  DRL4KDD'19. [paper](https://arxiv.org/pdf/1801.00209.pdf) [code(authority)](https://github.com/luozachary/drl-rec) [code](https://github.com/egipcy/LIRD)  
*employs **Actor-Critic** framework to learn the optimal strategy by a online simulator*  
*JD*

#### 2018
1. **Learning to Collaborate: Multi-Scenario Ranking via Multi-Agent Reinforcement Learning**. Jun Feng, Heng Li, Minlie Huang, Shichen Liu, Wenwu Ou, Zhirong Wang, Xiaoyan Zhu. WWW 2018. [paper](https://arxiv.org/pdf/1809.06260.pdf)  
*uses the multi-agent reinforcement learning to optimize the multi-scenario ranking.*

1. **Reinforcement Mechanism Design for e-commerce**. Qingpeng Cai, Aris Filos-Ratsikas, Pingzhong Tang, Yiwei Zhang. WWW 2018. [paper](https://arxiv.org/pdf/1708.07607.pdf)

1. **DRN: A Deep Reinforcement Learning Framework for News Recommendation**. Guanjie Zheng, Fuzheng Zhang, Zihan Zheng, Yang Xiang, Nicholas Jing Yuan, Xing Xie, Zhenhui Li. WWW 2018. [paper](http://www.personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)  
*Pennsylvania State University, Microsoft*

1. **Deep Reinforcement Learning for Page-wise Recommendations**. Xiangyu Zhao, Long Xia, Liang Zhang, Zhuoye Ding, Dawei Yin, Jiliang Tang.  RecSys 2018. [paper](https://arxiv.org/pdf/1805.02343.pdf)  
*Adopts RL to recommend items on a 2-D page instead of showing one single item each time.*  
*simulator used*

1. **Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning**. Xiangyu Zhao, Liang Zhang, Zhuoye Ding, Long Xia, Jiliang Tang, Dawei Yin. KDD 2018. [paper](https://arxiv.org/pdf/1802.06501.pdf)  
*considers both positive and negative feedback from users recent behaviors to help find optimal strategy.*  
*simulator used*

1. **Stabilizing Reinforcement Learning in Dynamic Environment with Application to Online Recommendation**. Shi-Yong Chen, Yang Yu, Qing Da, Jun Tan, Hai-Kuan Huang, Hai-Hong Tang. KDD 2018. [paper](http://lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/kdd18-RobustDQN.pdf)  
*To mitigate the performance degradation due to high-variance and biased estimation of the reward, the paper provides a stratified random sampling and an approximate regretted reward to enhance the robustness of the model.*

1. **Reinforcement Learning to Rank in E-Commerce Search Engine: Formalization, Analysis, and Application**. Yujing Hu, Qing Da, Anxiang Zeng, Yang Yu, Yinghui Xu. KDD 2018. [paper](https://arxiv.org/pdf/1803.00710.pdf)  
*introduces DPG-FBE algorithm to maintain an approximate model of the environment to perform reliable updates of value functions.*

1. **A Reinforcement Learning Framework for Explainable Recommendation**. Xiting Wang, Yiru Chen, Jie Yang, Le Wu, Zhengtao Wu, Xing Xie. ICDM 2018. [paper](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/main.pdf)

#### 2019

1. **Top-K Off-Policy Correction for a REINFORCE Recommender System**. Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed H. Chi. WSDM 2019. [paper](https://arxiv.org/pdf/1812.02353.pdf) [reproduce](https://github.com/mercurialgh/Reproduce-of-Top-K-Off-Policy-Correction-for-a-REINFORCE-Recommender-System)  
*Youtube*

1. **Generative Adversarial User Model for Reinforcement Learning Based Recommendation System**. Xinshi Chen, Shuang Li, Hui Li, Shaohua Jiang, Yuan Qi, Le Song. ICML 2019. [paper](http://proceedings.mlr.press/v97/chen19f/chen19f.pdf)

1. **Aggregating E-commerce Search Results from Heterogeneous Sources via Hierarchical Reinforcement Learning**. Ryuichi Takanobu, Tao Zhuang, Minlie Huang, Jun Feng, Haihong Tang, Bo Zheng. WWW 2019. [paper](https://arxiv.org/pdf/1902.08882.pdf)

1. **Policy Gradients for Contextual Recommendations**. Feiyang Pan, Qingpeng Cai, Pingzhong Tang, Fuzhen Zhuang, Qing He. WWW 2019. [paper](https://arxiv.org/pdf/1802.04162.pdf)

1. ★★★  
**Value-aware Recommendation based on Reinforcement Profit Maximization**. Changhua Pei, Xinru Yang, Qing Cui, Xiao Lin, Fei Sun, Peng Jiang, Wenwu Ou, Yongfeng Zhang. WWW 2019.  
*Value-aware*

1. **Reinforcement Knowledge Graph Reasoning for Explainable Recommendation**. Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang. SIGIR 2019. [paper](http://yongfeng.me/attach/xian-sigir2019.pdf)

1. **Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems**. Lixin Zou, Long Xia, Zhuoye Ding, Jiaxing Song, Weidong Liu, Dawei Yin. KDD 2019. [paper](https://arxiv.org/pdf/1902.05570.pdf)

1. **Environment reconstruction with hidden confounders for reinforcement learning based recommendation**. Wenjie Shang, Yang Yu, Qingyang Li, Zhiwei Qin, Yiping Meng, Jieping Ye. KDD 2019. [paper](http://lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/kdd19-confounder.pdf)

1. **Exact-K Recommendation via Maximal Clique Optimization**. Yu Gong, Yu Zhu, Lu Duan, Qingwen Liu, Ziyu Guan, Fei Sun, Wenwu Ou, Kenny Q. Zhu. KDD 2019. [paper](https://arxiv.org/pdf/1905.07089.pdf)

1. **Hierarchical Reinforcement Learning for Course Recommendation in MOOCs**. Jing Zhang, Bowen Hao, Bo Chen, Cuiping Li, Hong Chen, Jimeng Sun. AAAI 2019. [paper](https://xiaojingzi.github.io/publications/AAAI19-zhang-et-al-HRL.pdf)

1. **Large-scale Interactive Recommendation with Tree-structured Policy Gradient**. Haokun Chen, Xinyi Dai, Han Cai, Weinan Zhang, Xuejian Wang, Ruiming Tang, Yuzhou Zhang, Yong Yu. AAAI 2019. [paper](https://arxiv.org/pdf/1811.05869.pdf)

1. **Virtual-Taobao: Virtualizing real-world online retail environment for reinforcement learning**. Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, An-Xiang Zeng. AAAI 2019. [paper](http://www.lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/aaai2019-virtualtaobao.pdf)

1. **A Model-Based Reinforcement Learning with Adversarial Training for Online Recommendation**. Xueying Bai, Jian Guan, Hongning Wang. NeurIPS 2019. [paper](http://papers.nips.cc/paper/9257-a-model-based-reinforcement-learning-with-adversarial-training-for-online-recommendation.pdf)

1. **Text-Based Interactive Recommendation via Constraint-Augmented Reinforcement Learning**. Ruiyi Zhang, Tong Yu, Yilin Shen, Hongxia Jin, Changyou Chen, Lawrence Carin. NeurIPS 2019. [paper](http://people.ee.duke.edu/~lcarin/Ruiyi_NeurIPS2019.pdf)

1. **Model Based Reinforcement Learning for Online Recommendation**. . NeurIPS 2019. [paper](https://arxiv.org/pdf/1911.03845.pdf) [code](https://github.com/XueyingBai/Model-Based-Reinforcement-Learning-for-Online-Recommendation)

1. **DRCGR: Deep reinforcement learning framework incorporating CNN and GAN-based for interactive recommendation**. Rong Gao, Haifeng Xia, Jing Li, Donghua Liu, Shuai Chen, and Gang Chun. ICDM 2019. [paper](https://ieeexplore.ieee.org/document/8970700)

#### 2020

1. **Pseudo Dyna-Q: A Reinforcement Learning Framework for Interactive Recommendation**. Lixin Zou, Long Xia, Pan Du, Zhuo Zhang, Ting Bai, Weidong Liu, Jian-Yun Nie, Dawei Yin. WSDM 2020. [paper](https://tbbaby.github.io/pub/wsdm20.pdf)  
*Hui SHI lectured*  

1. **End-to-End Deep Reinforcement Learning based Recommendation with Supervised Embedding**. Feng Liu, Huifeng Guo, Xutao Li, Ruiming Tang, Yunming Ye, Xiuqiang He. WSDM 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371858)

1. **Reinforced Negative Sampling over Knowledge Graph for Recommendation**. Xiang Wang, Yaokun Xu, Xiangnan He, Yixin Cao, Meng Wang, Tat-Seng Chua. WWW 2020. [paper](https://arxiv.org/pdf/2003.05753.pdf)

1. **A Reinforcement Learning Framework for Relevance Feedback**. Ali Montazeralghaem, Hamed Zamani, James Allan. SIGIR 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401099)

1. ★★★  
**KERL: A Knowledge-Guided Reinforcement Learning Model for  Sequential Recommendation**. Pengfei Wang, Yu Fan, Long Xia, Wayne Xin Zhao, Shaozhang Niu, Jimmy Huang. SIGIR 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401134) [code](https://github.com/fanyubupt/KERL)  
*Graph*

1. **Self-Supervised Reinforcement Learning for Recommender Systems**. Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, Joemon Jose. SIGIR 2020. [paper](https://arxiv.org/pdf/2006.05779.pdf)  
*Self-Supervised*  

1. **Reinforcement Learning to Rank with Pairwise Policy Gradient**. Jun Xu, Zeng Wei, Long Xia, Yanyan Lan, Dawei Yin, Xueqi Cheng, Ji-Rong Wen. SIGIR 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401148)

1. ★★★  
**MaHRL: Multi-goals Abstraction based Deep Hierarchical Reinforcement Learning for Recommendations**. Dongyang Zhao, Liang Zhang, Bo Zhang, Lizhou Zheng, Yongjun Bao, Weipeng Yan. SIGIR 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401170)  
(Deep Hierarchical Reinforcement Learning Based Recommendations via Multi-goals Abstraction. In Proceedings of SIGKDD 2019. [paper](https://arxiv.org/pdf/1903.09374.pdf) )  
*PKU / JD*  
*high-level agent and low-level agent*  
*Hierarchical*

1. **Leveraging Demonstrations for Reinforcement Recommendation Reasoning over Knowledge Graphs**. Kangzhi Zhao, Xiting Wang, Yuren Zhang, Li Zhao, Zheng Liu, Chunxiao Xing, Xing Xie. SIGIR 2020. [paper](https://dl.acm.org/doi/10.1145/3397271.3401171)

1. **Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning**. Sijin Zhou, Xinyi Dai, Haokun Chen, Weinan Zhang, Kan Ren, Ruiming Tang, Xiuqiang He, Yong Yu. SIGIR 2020. [paper](https://arxiv.org/pdf/2006.10389.pdf)

1. **Adversarial Attack and Detection on Reinforcement Learning based Recommendation System**. Yuanjiang Cao, Xiaocong Chen, Lina Yao, Xianzhi Wang, Wei Emma Zhang. SIGIR 2020. [paper](https://arxiv.org/pdf/2006.07934.pdf)

1. **Reinforcement Learning based Recommendation with Graph Convolutional Q-network**. Yu Lei, Hongbin Pei, Hanqi Yan, Wenjie Li. SIGIR 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401237)

1. **Nonintrusive-Sensing and Reinforcement-Learning Based Adaptive Personalized Music Recommendation**. D Hong, L Miao, Y Li. SIGIR 2020. [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401225)

1. **Joint Policy-Value Learning for Recommendation**. Olivier Jeunen, David Rohde, Flavian Vasile, Martin Bompaire. KDD 2020. [link(with video)](https://www.kdd.org/kdd2020/accepted-papers/view/joint-policy-value-learning-for-recommendation)  [paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403175)  [video](https://www.youtube.com/watch?v=fMwxxbcXk8c)  
*Combine value learning and policy learning*

1. **Keeping Dataset Biases out of the Simulation: A Debiased Simulator for Reinforcement Learning based Recommender Systems**. Jin Huang, Harrie Oosterhuis, Maarten de Rijke, Herke van Hoof. RecSys 2020. [paper](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/huang-2020-keeping.pdf)

1. **Learning to Collaborate in Multi-Module Recommendation via Multi-Agent Reinforcement Learning without Communication**. Xu He, Bo An, Yanghua Li, Haikai Chen, Rundong Wang, Xinrun Wang, Runsheng Yu, Xin Li, Zhirong Wang. RecSys 2020. [paper](https://arxiv.org/pdf/2008.09369.pdf)
1. **User Response Models to Improve a REINFORCE Recommender System**. Minmin Chen, Bo Chang, Can Xu, Ed Chi. WSDM 2021. [paper](https://dl.acm.org/doi/pdf/10.1145/3437963.3441764)

#### 2021

1. **Hierarchical Reinforcement Learning for Integrated Recommendation**. Ruobing Xie, Shaoliang Zhang, Rui Wang, Feng Xia, Leyu Lin. AAAI, 2021. [paper](https://www.aaai.org/AAAI21Papers/AAAI-1169.XieR.pdf)  
*Hui SHI lectured*  
*WeChat*  
*Base MaHRL*
*high-level agent and low-level agent*  
*Hierarchical*  

1. **Cost-Effective and Interpretable Job Skill Recommendation with Deep Reinforcement Learning.** Ying Sun, Fuzhen Zhuang, Hengshu Zhu, Qing He, Hui Xiong. WWW 2021. [paper](https://dl.acm.org/doi/10.1145/3442381.3449985) [video](https://www.youtube.com/watch?v=PFurYK0mwnE)  
*Hui Xiong*  
*Multi-task RL for Rec*

### Preprint Papers
1. **Reinforcement Learning based Recommender System using Biclustering Technique**. Sungwoon Choi, Heonseok Ha, Uiwon Hwang, Chanju Kim, Jung-Woo Ha, Sungroh Yoon. arxiv 2018. [paper](https://arxiv.org/pdf/1801.05532.pdf) 

1. **Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling**. Feng Liu, Ruiming Tang, Xutao Li, Weinan Zhang, Yunming Ye, Haokun Chen, Huifeng Guo, Yuzhou Zhang. arxiv 2018. [paper](https://arxiv.org/pdf/1810.12027.pdf)

1. **Model-Based Reinforcement Learning for Whole-Chain Recommendations**. Xiangyu Zhao, Long Xia, Yihong Zhao, Dawei Yin, Jiliang Tang. arxiv 2019. [paper](https://arxiv.org/pdf/1902.03987.pdf)

### Accepted Paper List of Top Conference 
**KDD 2019**  
https://dblp.org/db/conf/kdd/kdd2019.html

**KDD 2020**  
https://www.kdd.org/kdd2020/accepted-papers

**RecSYS 2020**  
https://recsys.acm.org/recsys20/accepted-contributions/

**Sigir 2020**  
https://sigir.org/sigir2020/accepted-papers/

**WWW 2021**  
https://www2021.thewebconf.org/program/papers/