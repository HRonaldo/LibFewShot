# 小样本学习及论文

## 小样本学习(Few-Shot Learining FSL)
+ <https://zhuanlan.zhihu.com/p/258562899>
- Introduction to Few-Shot Learning
  - 数学表述：
    - $D_T = (D_{trn},D_{tst})$ $D_ {trn} =\left \{ {( x_ {i} , y_ {i} )} \right \}  _ {i=1}^ {N_ {trn}}$  $D_{tst} = \left \{ x_j \right \}$ 并且有$x_i,x_j \in X_r\subset X,y_i\in Y_r\subset Y$
    如果$D_{trn}$ 中有c个类，每个类有K个样本(K非常小)，那么这个任务就是c-way K-shot分类任务。
    - 单纯使用训练集和测试集，效果较差，故一般情况下会使用一个有监督的辅助数据集，记作
    $D_A=(x_i^a,y_i^a)_i^{N_{out}} ,x_i^a\in X_A\subset X,y_i^a\in Y_A\subset Y$
    - 需要注意的是D_A中不包含任务D_T中的样本类别，即$Y_A\cap Y_r=\emptyset$(称为类别正交)，否则就不是小样本学习
- Two types of Few-Shot Learning
  + <https://www.zhihu.com/question/20446337>
  - 判别式模型:后验概率(SVM)
  - 生成式模型:联合概率(高斯混合分布)
  + 元学习<https://zhuanlan.zhihu.com/p/136975128> 学习如何学习
    - 第一层训练单位是任务，也就是说，元学习中要准备许多任务来进行学习，第二层训练单位才是每个任务对应的数据

## Interventional Few-shot Learning
### 相关知识
+ Fine-tuning:微调
+ 结构因果模型(SCM):变量被表示为节点，因果关系被表示为有向边,用于描述和推断因果关系的一种数学框架<https://zhuanlan.zhihu.com/p/33860572>
### abstract
- Viewpoint:预训练知识实际上是一个限制性能的混淆因素
- 因果假设：预训练的知识、样本特征和标签之间的因果关系的结构因果模型(SCM)





## Code Analysis
- MAML_MN_FT(PyTorch)
- MTL(PyTorch)
- SIB(PyTorch)



## Reference
因果推理与ML结合
- <https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/119950077?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-119950077-blog-111514124.235^v43^pc_blog_bottom_relevance_base5&spm=1001.2101.3001.4242.1&utm_relevant_index=3>

因果推理相关知识
- <https://zhuanlan.zhihu.com/p/111306353>
- <https://www.zhihu.com/column/c_1217887302124773376>