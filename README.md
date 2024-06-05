# knowgraph 


## 1. 简介

https://github.com/malllabiisc/CompGCN

多元关系知识表示学习进展
https://zhuanlan.zhihu.com/p/570314075

https://github.com/migalkin/StarE

StarE中C_models、run-StarE放入上级目录即可


常用命令
```
# git
$ cd e:/code/Git/knowgraph
$ git add .
$ git status
$ git commit -m "change"
$ git pull origin master
$ git push origin master

# python
conda activate pytorch
cd /home/GXW/code/pytorch/knowgraph
cd /home/GXW/code/pytorch/knowgraph/D_export/log
cd /home/GXW/code/pytorch/knowgraph/B_data/knowgraph

# 不使用测试集
nohup python -u ./run-notest.py > /dev/null 2>&1 &
kill -9 $(pgrep -f './run-notest.py')
nohup python -u ./run-StarE.py >> ./run-StarE.log 2>&1 &
kill -9 $(pgrep -f './run-StarE.py')
nohup python -u datadeal-allDD-notest.py >> ./run.log 2>&1 &
kill -9 $(pgrep -f 'datadeal-allDD-notest.py')
nohup python -u RawAccdeal-BERT.py >> ./BERT_max_run.log 2>&1 &
kill -9 $(pgrep -f 'RawAccdeal-BERT.py')

# 使用测试集
nohup python -u ./run.py > /dev/null 2>&1 &
kill -9 $(pgrep -f './run.py')
nohup python -u datadeal.py >> ./run.log 2>&1 &
kill -9 $(pgrep -f 'datadeal.py')
```

## 2. 调参

### 2.1. 重采样

聊聊Pytorch中的dataloader——sampler模块
https://zhuanlan.zhihu.com/p/117270644

当dataloader的shuffle参数为True时，系统会自动调用采样器data.RandomSampler——数据随机采样
同一个epoch下的batch中有重复数据
replacement用于指定是否可以重复选取某一个样本，默认为True，即允许在一个epoch中重复采样某一个数据
https://blog.csdn.net/qq_50001789/article/details/128974424

### 2.2. 学习率

调整学习率（torch.optim.lr_scheduler）
https://blog.csdn.net/qq_40206371/article/details/119910592
基于FFT计算循环相关的内容
https://blog.csdn.net/qq_51698536/article/details/131843669


### 3. 论文
基于知识图谱的图神经网络推理
https://zhuanlan.zhihu.com/p/647114494
CompGCN
https://blog.csdn.net/sinat_28978363/article/details/105298286