# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 16:17
# @Author  : Ryan
# @PRO_NAME: PyTorchLearning
# @File    : optuna_yolov7.py
# @Software: PyCharm 
# @Comment :
import torch
import optuna
from pathlib import Path
import subprocess

# 定义训练函数
def train_yolov7(trial):
    # 定义超参数搜索空间
    hyp = {
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'momentum': trial.suggest_uniform('momentum', 0.0, 1.0),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
        'epochs': trial.suggest_int('epochs', 100, 200)
        # 添加更多的超参数...
    }

    # 将超参数写入YOLOv7的超参数配置文件
    with open('./data/hyp.scratch.p5.yaml', 'w') as f:
        for k, v in hyp.items():
            f.write(f'{k}: {v}\n')

    # 执行YOLOv7的训练命令
    process = subprocess.Popen(['python', 'train.py', '--img', '416', '--batch', '128', '--epochs', '200'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # 解析训练结果，例如获取验证集上的mAP
    # 这里以获取最后一个epoch的验证集mAP为例
    result = float(stderr.decode().splitlines()[-1].split(':')[-1].strip())

    return result

# 定义Optuna的目标函数
def objective(trial):
    # 调用训练函数并返回结果
    result = train_yolov7(trial)

    # 保存每个试验的训练结果
    trial.set_user_attr('result', result)

    return result

# 创建Optuna的Study对象并运行优化
study = optuna.create_study(direction='maximize')  # 创建Study对象，指定优化目标是最大化
study.optimize(objective, n_trials=100)  # 执行优化，n_trials指定优化的迭代次数

# 打印优化结果
best_trial = study.best_trial
print('Best trial:')
print(f'  Value: {best_trial.value}')
print('  Params: ')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')

# 使用最佳超参数进行额外的操作
best_params = best_trial.params
# 在这里执行使用最佳超参数的操作，例如再次训练模型、进行推理等