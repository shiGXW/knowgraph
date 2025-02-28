# knowgraph 


## 1. 简介

https://github.com/malllabiisc/CompGCN

多元关系知识表示学习进展
https://zhuanlan.zhihu.com/p/570314075

https://github.com/migalkin/StarE

StarE中C_models、run-StarE放入上级目录即可

Pytorch中使用TensorBoard
https://blog.csdn.net/m0_61878383/article/details/136552258


常用命令
```
# git
$ cd e:/code/Git/knowgraph
$ git add .
$ git status
$ git commit -m "change"
$ git pull origin master
$ git push origin master
$ git pull origin_hub master
$ git push origin_hub master

# python
conda activate pytorch

cd /home/GXW/code/pytorch/knowgraph
cd /home/GXW/code/pytorch/knowgraph/D_export/log
cd /home/GXW/code/pytorch/knowgraph/B_data/knowgraph

cd /root/GXM/code/knowgraph
cd /root/GXM/code/knowgraph/D_export/log
cd /root/GXM/code/knowgraph/D_export/checkpoint
cd /root/GXM/code/knowgraph/B_data/knowgraph/BERT
cat RawAccdeal_BERT_run.log
cd /root/GXM/code/knowgraph/B_data/datasets/knowgraph/maxDDD
cat rawacc_beone_max_dict_simple.json
cd /root/GXM/code/knowgraph/B_data/knowgraph/datasetPart
cat datadeal_all_run.log
cd /root/GXM/code/knowgraph/B_data/knowgraph/Neo
cat data2neo4jDDD_run.log

# 不使用测试集
# 模型训练+验证
nohup python -u ./run-notest.py > /dev/null 2>&1 &
kill -9 $(pgrep -f './run-notest.py')
nohup python -u ./run-StarE.py >> ./run-StarE.log 2>&1 &
kill -9 $(pgrep -f './run-StarE.py')
# 模型验证
nohup python -u ./run-notest.py -restore True -name testrun_2025_01_08_17_23_40 > /dev/null 2>&1 &
kill -9 $(pgrep -f './run-notest.py')
# 实体链接
nohup python -u RawAccdeal-BERTDDD-simple.py >> ./RawAccdeal_BERT_run.log 2>&1 &
kill -9 $(pgrep -f 'RawAccdeal-BERTDDD-simple.py')
# 数据处理
nohup python -u datadeal-allDDD-notest.py >> ./datadeal_all_run.log 2>&1 &
kill -9 $(pgrep -f 'datadeal-allDDD-notest.py')
# 导入neo4j
nohup python -u data2neo4jDDD.py >> ./data2neo4jDDD_run.log 2>&1 &
kill -9 $(pgrep -f 'data2neo4jDDD.py')

# 使用测试集
nohup python -u ./run.py > /dev/null 2>&1 &
kill -9 $(pgrep -f './run.py')
nohup python -u datadeal.py >> ./run.log 2>&1 &
kill -9 $(pgrep -f 'datadeal.py')

# TensorBoard 训练过程可视化
cd testrun_2024_07_20_09_56_31_csv
tensorboard --logdir=./ --bind_all
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