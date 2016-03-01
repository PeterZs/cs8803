#!/bin/bash

qlua run.lua --model MyLinear --optMethod sgd
mv results/train.log.eps results/sgd_train.eps
mv results/test.log.eps results/sgd_test.eps
qlua run.lua --model MyLinear --optMethod sgd -r 1e-2
mv results/train.log.eps results/sgd_r2_train.eps
mv results/test.log.eps results/sgd_r2_test.eps
qlua run.lua --model MyLinear --optMethod sgd -r 1e-4
mv results/train.log.eps results/sgd_r4_train.eps
mv results/test.log.eps results/sgd_r4_test.eps
qlua run.lua --model MyLinear --optMethod sgd -d 1e-5
mv results/train.log.eps results/sgd_d5_train.eps
mv results/test.log.eps results/sgd_d5_test.eps
qlua run.lua --model MyLinear --optMethod sgd -d 1e-6
mv results/train.log.eps results/sgd_d6_train.eps
mv results/test.log.eps results/sgd_d6_test.eps
qlua run.lua --model MyLinear --optMethod sgd_w
mv results/train.log.eps results/sgdw_train.eps
mv results/test.log.eps results/sgdw_test.eps
qlua run.lua --model MyLinear --optMethod sgd_w -w 1e-3
mv results/train.log.eps results/sgdw_w3_train.eps
mv results/test.log.eps results/sgdw_w3_test.eps
qlua run.lua --model MyLinear --optMethod sgd_w -w 1e-4
mv results/train.log.eps results/sgdw_w4_train.eps
mv results/test.log.eps results/sgdw_w4_test.eps
qlua run.lua --model MyLinear --optMethod sgd_wm
mv results/train.log.eps results/sgdwm_train.eps
mv results/test.log.eps results/sgdwm_test.eps
qlua run.lua --model MyLinear --optMethod sgd_wm -m 0.5
mv results/train.log.eps results/sgdwm_m5_train.eps
mv results/test.log.eps results/sgdwm_m5_test.eps
qlua run.lua --model MyLinear --optMethod sgd_wm -m 0.9
mv results/train.log.eps results/sgdwm_m9_train.eps
mv results/test.log.eps results/sgdwm_m9_test.eps
qlua run.lua --model MyLinear --learningRate 1e-4 --learningRateDecay 1e-7 --batchSize 2 --weightDecay 1e-5 --optMethod cd
mv results/train.log.eps results/cd_train.eps
mv results/test.log.eps results/cd_test.eps
qlua run.lua --model MyLinear --learningRate 1e-3 --learningRateDecay 1e-7 --batchSize 2 --weightDecay 1e-5 --optMethod cd
mv results/train.log.eps results/cd_lr3_train.eps
mv results/test.log.eps results/cd_lr3_test.eps
qlua run.lua --model MyLinear --learningRate 1e-5 --learningRateDecay 1e-7 --batchSize 2 --weightDecay 1e-5 --optMethod cd
mv results/train.log.eps results/cd_lr5_train.eps
mv results/test.log.eps results/cd_lr5_test.eps

qlua run.lua --model mlp_tanh
mv results/train.log.eps results/tanh_train.eps
mv results/test.log.eps results/tanh_test.eps
qlua run.lua --model mlp_sigmoid
mv results/train.log.eps results/sigmoid_train.eps
mv results/test.log.eps results/sigmoid_test.eps
qlua run.lua --model mlp_relu
mv results/train.log.eps results/relu_train.eps
mv results/test.log.eps results/relu_test.eps
qlua run.lua --model mlp_requ
mv results/train.log.eps results/requ_train.eps
mv results/test.log.eps results/requ_test.eps
