**这个文件夹内的程序用来提取RNA或者DNA特征**

```
运行命令如下：
	python program.py input.fasta output.csv gene_type whether_N_filled
```
program.py: 代表具体的程序，可在此文件夹内找到

input.fasta: 代表输入的fasta文件

output.csv: 代表输出的csv文件

gene_type: 可选参数是 RNA或者DNA

whether_N_filled: 代表的是fasta文件内是否有N填充空白位置的存在,1代表存在，0代表不存在

note: 输出的csv文件不带有标签，因此输入的文件最好是平衡数据集，这样前一半是正集合，后一半是负集合，比较好区分
	`
example: 
	python RFH.py S3_Athaliana.fasta output.csv DNA 0
`
