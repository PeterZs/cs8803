#!/bin/bash

qlua run.lua --model MyLinear --optMethod sgd
mv results/train.log.eps results/sgd_train.log.eps
mv results/test.log.eps results/sgd_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd -r 1e-2
mv results/train.log.eps results/sgd_r2_train.log.eps
mv results/test.log.eps results/sgd_r2_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd -r 1e-4
mv results/train.log.eps results/sgd_r4_train.log.eps
mv results/test.log.eps results/sgd_r4_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd -d 1e-5
mv results/train.log.eps results/sgd_d5_train.log.eps
mv results/test.log.eps results/sgd_d5_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd -d 1e-6
mv results/train.log.eps results/sgd_d6_train.log.eps
mv results/test.log.eps results/sgd_d6_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd_w
mv results/train.log.eps results/sgdw_train.log.eps
mv results/test.log.eps results/sgdw_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd_w -w 1e-3
mv results/train.log.eps results/sgdw_w3_train.log.eps
mv results/test.log.eps results/sgdw_w3_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd_w -w 1e-4
mv results/train.log.eps results/sgdw_w4_train.log.eps
mv results/test.log.eps results/sgdw_w4_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd_wm
mv results/train.log.eps results/sgdwm_train.log.eps
mv results/test.log.eps results/sgdwm_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd_wm -m 0.5
mv results/train.log.eps results/sgdwm_m5_train.log.eps
mv results/test.log.eps results/sgdwm_m5_test.log.eps
qlua run.lua --model MyLinear --optMethod sgd_wm -m 0.9
mv results/train.log.eps results/sgdwm_m9_train.log.eps
mv results/test.log.eps results/sgdwm_m9_test.log.eps
qlua run.lua --model MyLinear --learningRate 1e-4 --learningRateDecay 1e-7 --batchSize 2 --weightDecay 1e-5 --optMethod cd
mv results/train.log.eps results/cd_train.log.eps
mv results/test.log.eps results/cd_test.log.eps
qlua run.lua --model MyLinear --learningRate 1e-3 --learningRateDecay 1e-7 --batchSize 2 --weightDecay 1e-5 --optMethod cd
mv results/train.log.eps results/cd_lr3_train.log.eps
mv results/test.log.eps results/cd_lr3_test.log.eps
qlua run.lua --model MyLinear --learningRate 1e-5 --learningRateDecay 1e-7 --batchSize 2 --weightDecay 1e-5 --optMethod cd
mv results/train.log.eps results/cd_lr5_train.log.eps
mv results/test.log.eps results/cd_lr5_test.log.eps

qlua run.lua --model mlp_tanh
mv results/train.log.eps results/tanh_train.log.eps
mv results/test.log.eps results/tanh_test.log.eps
qlua run.lua --model mlp_sigmoid
mv results/train.log.eps results/sigmoid_train.log.eps
mv results/test.log.eps results/sigmoid_test.log.eps
qlua run.lua --model mlp_relu
mv results/train.log.eps results/relu_train.log.eps
mv results/test.log.eps results/relu_test.log.eps
qlua run.lua --model mlp_requ
mv results/train.log.eps results/requ_train.log.eps
mv results/test.log.eps results/requ_test.log.eps
