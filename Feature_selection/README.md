**此文件夹用来存放特征选择程序**
因为一般的特征选择程序都会在scikit-learn或者99服务器上，调用就行。
因此，在这边我们只存放一些调用程序，或者类似半Pipeline的方式。

```
	python program.py input.csv crossvalidation_value, n_jobs_value
```
```
program.py: 代表程序

input.csv: 代表特征提取完成的文件，

crossvalidation_value: 代表交叉验证的折数

n_jobs_value: 代表并行的CPU数量，(-1)表示为占用全部CPU,因此不推荐，thinkpad
一般是4核心，因此为了不卡顿填3， 131或者132服务器上可以填8或12(一定不能填-1不然会被人打)
```

**依赖**
* python 2.7.x
* pandas
* scikit-learn
* xlwt
* sys
* itertools
* xgboost
* math
* subprocess
* numpy
* warnings
* easy_excel (这是附加的一个程序包，万师兄写的，我之前刚刚学习python的时候图方便调用的，其实pandas有更好的方法实现，只是我懒得改了，因此一直调用着)[链接](https://github.com/ShixiangWan/Easy-Classify)
