**PSKNP 加间隔gap**

```
	python program.py input.fasta K_value gap_value crossvalidation n_jobs positive_samples_num negative_samples_num
```
```
program.py: 这是python的代码程序
input.fasta: 这是输入的RNA或者DNA序列文件（fasta格式）
K_value: 几个核苷酸的列频率统计
gap_value: 核苷酸的跳跃间隔
crossvalidation: 几折交叉验证，如 10 代表十折交叉验证
n_jobs: 开多少cpu核心进行并行操作，若填-1则代表全部用，但是不推荐这样，因为全部占用会导致电脑卡顿做不了其他事情，尤其不能在共享的服务器上用，或者会他人资源，
positive_samples_num： 正集合的数量
negative_samples_num：负集合的数量
```
```
	example: python PSKNP.py saccgaromyces_cerevisiae_dataset.fasta 3 0 10 3 10 10
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
