**这个文件夹用来存放那些不能特征提取跟分类器训练分开的，或者需要一步到位出结果的程序**

```
	python program.py input.fasta gene_type whether_N_filled crossvalidation n_jobs
```
```
program.py: 这是python的代码程序
input.fasta: 这是输入的RNA或者DNA序列文件（fasta格式）（平衡数据集）
gene_type: RNA或者DNA
whether_N_filled: 是否N填充，若为N填充则填1，否则为0
crossvalidation: 几折交叉验证，如 10 代表十折交叉验证
n_jobs: 开多少cpu核心进行并行操作，若填-1则代表全部用，但是不推荐这样，因为全部占用会导致电脑卡顿做不了其他事情，尤其不能在共享的服务器上用，或者会他人资源，
```
```
	example: python PSNP.py saccgaromyces_cerevisiae_dataset.fasta RNA 0 10 2
```
**依赖**
* python2.7.x
* pandas
* sklearn
* itertools
* math
* sys
* warnings
* easy_excel (这个包乃是我万师兄做的，我程序用调用)[链接地址](https://github.com/ShixiangWan/Easy-Classify)
