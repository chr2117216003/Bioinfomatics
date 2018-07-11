**这个文件夹内的程序用来参数优化和训练模型，最后得到交叉验证的结果**

**涉及到的分类器有:**
* Support Vector Machine(SVM)
* K-Nearest Neighbors(KNN)
* Random Forest(RF)
* Naive Bayes(NB)
* Logistic Regression(LR)
* Gradient Boosting Decision Tree
* eXtreme Gradient Boosting(XGBoost) 
```
	python program.py input.csv crossvalidation_value n_jobs_value
```

```
	program.py: 代表程序(其中easy_excel.py不是使用的程序，不可用，只是辅助的工具)
	
	input.csv: 代表通过特征提取完成后的csv文件，文件中不应含有标签，默认前一半为正集合，后一半为负集合(平衡数据集)
	
	crossvalidation_value: 代表交叉验证的折数
	
	n_jobs_value: 代表并行的CPU数，希望运行的程序不要把电脑或者服务器的cpu都占满，一般电脑4个cpu因此可以填3,131或者132服务器可以填12.(-1代表cpu全用，不推荐)
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
* easy_excel (这是附加的一个程序包，万师兄写的，我之前刚刚学习python的时候图方便调用的，其实pandas有更好的方法实现，只是我懒得改了，因此一直调用着)[链接](https://github.com/ShixiangWan/Easy-Classify)

**附**
·这里面的分类器都是经过参数寻优的，因此训练的非常慢，尤其在上千个的数据集中更慢，若为了图方便快捷一点，或者在论文的第一个分类器比较中不需要优化精度的比较，可以考虑万世想师兄的[Easy-Classify](https://github.com/ShixiangWan/Easy-Classify)
·
