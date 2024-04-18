# knowgraph 


## 1. 简介

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
cd /home/GXW/code/pytorch/knowgraph/B_data

# 不使用测试集
nohup python -u ./run-notest.py > /dev/null 2>&1 &
kill -9 $(pgrep -f './run-notest.py')
nohup python -u datadeal-notest.py >> ./run-new.log 2>&1 &
kill -9 $(pgrep -f 'datadeal-notest.py')

# 使用测试集
nohup python -u ./run.py > /dev/null 2>&1 &
kill -9 $(pgrep -f './run.py')
nohup python -u datadeal.py >> ./run.log 2>&1 &
kill -9 $(pgrep -f 'datadeal.py')
```

## 2.