#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
	parser = argparse.ArgumentParser()
	# save file
	parser.add_argument('--save', type=str, default='save',
	                    help="dic to save results (ending without /)")
	
	parser.add_argument('--init', type=str, default='None',
	                    help="location of init model")
	
	# federated arguments
	parser.add_argument('--epochs', type=int, default=300,
	                    help="rounds of training")
	
	parser.add_argument('--num_users', type=int,
	                    default=1000, help="number of users: K")
	
	parser.add_argument('--frac', type=float, default=0.1,
	                    help="the fraction of clients: C")
	
	parser.add_argument('--malicious', type=float, default=0, help="proportion of malicious clients")#0.04
	
	# ***** badnet labelflip layerattack updateflip get_weight layerattack_rev layerattack_ER adaptive****
	parser.add_argument('--attack', type=str,
	                    default='biasattack', help='attack method',  # 还有一个MR攻击
	                    choices=['biasattack, dba, adaptive, labelflip, biasattack, layerattack, flipupdate'])
	
	parser.add_argument('--ada_mode', type=int, default=1, help='0 denotes True, 1 denotes False,'
	                                                            'adaptive attack mode')
	
	parser.add_argument('--tau', type=float, default=0.8, help="threshold of LPA_ER")
	
	parser.add_argument('--poison_frac', type=float, default=0.3,
	                    help="fraction of dataset to corrupt for backdoor attack, 1.0 for layer attack")
	
	# *****local_ep = 3, local_bs=50, lr=0.1*******
	parser.add_argument('--local_ep', type=int, default=1,
	                    help="the number of local epochs: E")
	
	parser.add_argument('--local_bs', type=int, default=128,
	                    help="local batch size: B")
	
	parser.add_argument('--bs', type=int, default=128, help="test batch size")
	
	parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
	
	# model arguments
	# *************************model******************************#
	# resnet cnn VGG mlp Mnist_2NN Mnist_CNN resnet20 rlr_mnist
	parser.add_argument('--model', type=str,
	                    default='preactresnet18', help='model name',
	                    choices=['rlr_mnist, cnn, vgg19,resnet, preactresnet18, loannet'])
	
	# other arguments
	# *************************dataset*******************************#
	# fashion_mnist mnist cifar
	parser.add_argument('--dataset', type=str,
	                    default='cifar100', help="name of dataset",choices=['mnist','fmnist','gtsrb',
	                                                                     'cifar','cifar100','loan'])
	
	# ****0-avg, 1-fltrust 2-tr-mean 3-median 4-krum 5-muli_krum 6-RLR fltrust_bn fltrust_bn_lr****#
	parser.add_argument('--defence', type=str,
	                    default='avg', help="strategy of defence",
	                    choices=['avg,krum,multi_krum,median,flame,fltrust,RLR,fldetector,felguard,'
	                             'repeated_median', 'irls', 'irls_reputation','rocba','trimmed_mean'])
	
	parser.add_argument('--Lambda', type=float, default=2.0, help='set lambda of irls (default: 2.0)')
	parser.add_argument('--thresh', type=float, default=0.1, help='set thresh of irls restriction (default: 0.1)')
	parser.add_argument('--delta', type=float, default=0.2, help='set thresh of trimmed mean (default: 0.2)')
	
	parser.add_argument('--iid', type=int, default=1, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
	
	parser.add_argument('--alpha', type=float, default=0.05,
	                    help="the hyper-parameter that controls the gap between Euclidean distance and Cosine distance")
	
	parser.add_argument('--alpha_tri', type=float, default=0.2, help='set thresh of trimmed mean (default: 0.2)')
	
	parser.add_argument('--beta', type=int, default=5, help="the constant in non-linear mapping function")
	
	parser.add_argument('--k', type=int, default=2, help="parameter of krum")
	
	# parser.add_argument('--iid', action='store_true',
	#                     help='whether i.i.d or not')
	
	# parser.add_argument('--iid', type=int, default=1,
	#                     help='whether i.i.d (denoted as 1) or not (denoted as 0)')
	
	# ************************atttack_label********************************#
	parser.add_argument('--is_backdoor', type=bool, default=True, help="use backdoor attack")
	
	parser.add_argument('--attack_label', type=int, default=5, help="trigger for which label")
	
	# parser.add_argument('--single', type=int, default=-1,
	#                     help="single shot (denoted as 1) or repeated (denoted as 0),"
	#                          " when single=1, it indicates that performs the model replace attack")
	
	parser.add_argument('--backdoor_single_shot_scale_epoch', type=int, default=1,
	                    help="used for one-single-shot; -1 means no single scaled shot")
	
	# attack_goal=-1 is all to one
	
	parser.add_argument('--attack_goal', type=int, default=7, help="trigger to which label")
	
	# --attack_begin 70 means accuracy is up to 70 then attack
	parser.add_argument('--attack_begin', type=int, default=0, help="the accuracy begin to attack")
	
	# search times
	parser.add_argument('--search_times', type=int, default=20, help="binary search times")
	
	parser.add_argument('--gpu', type=int, default=7, help="GPU ID, -1 for CPU")
	
	parser.add_argument('--robustLR_threshold', type=int, default=4,
	                    help="break ties when votes sum to 0")
	
	parser.add_argument('--server_dataset', type=int, default=200, help="number of dataset in server")
	
	parser.add_argument('--server_lr', type=float, default=1, help="number of dataset in server using in fltrust")
	
	parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
	
	parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
	
	# our parameters
	parser.add_argument('--w_alpha', type=float, default=0.8, required=False,
	                    help='the aging weights of the positive reputation')
	
	parser.add_argument('--w_beta', type=float, default=0.2, required=False,
	                    help='the aging weights of the negative reputation')
	
	parser.add_argument('--alpha_i', type=float, default=0.8, required=False,
	                    help='the positive behavior of client i')
	
	parser.add_argument('--beta_i', type=float, default=0.2, required=False,
	                    help='the negative behavior of client i')
	
	parser.add_argument('--p_star', type=float, default=0.8, required=False,
	                    help='the weight of positive behavior of client i')
	
	parser.add_argument('--q_star', type=float, default=0.2, required=False,
	                    help='the weight of negative behavior of client i')
	
	parser.add_argument('--rho', type=int, default=30, required=False,
	                    help='the constant rho>=1 determines how quickly the modified tanh function changes around 0')
	
	parser.add_argument('--p', type=int, default=160, required=False,
	                    help='positive behavior that receives the global model')
	
	parser.add_argument('--q', type=int, default=40, required=False,
	                    help='negative behavior that rejects the global model')
	
	parser.add_argument('--reputation_active', type=bool, default=False, required=False,
	                    help='whether to use our reputation model')
	
	parser.add_argument('--reputation_effect', default=[], required=False,
	                    help='reputation array to store history of reputation scores')
	
	parser.add_argument('--loss_diff', type=float, default=0.00125, required=False, help='initial loss value')
	
	# NDSS那篇论文中需要的参数
	parser.add_argument('--kappa', type=float, default=0.3, required=False,
	                    help='weight for positive observation of the objective function '
	                         'applied to model when we consider R and S')
	
	parser.add_argument('--eta', type=float, default=0.7, required=False, help='eta + k = 1')
	
	parser.add_argument('--W', type=float, default=2, required=False,
	                    help='non-information prior weight is the weight of the reputation we assigned initially, w=2 assigned in paper by default')
	
	parser.add_argument('--a', type=float, default=0.8, required=False,
	                    help='a priori probability in the absence of committed belief mass. If we increase a, we scale up reputation')
	
	parser.add_argument('--z', type=float, default=0.6, required=False,
	                    help='time decay or interaction freshness z (0,1). If we increase z we scale up our reputation model')
	
	parser.add_argument('--s', type=float, default=10, required=False, help='window length')
	
	parser.add_argument('--reputation_active_type', type=int, default=0, choices=[0, 1], required=False,
	                    help='choose type of reputation model'
	                         '0: stands for Subjective logic')
	
	parser.add_argument('--reputation_timestamp', type=int, default=0, help='the reputation timestamp')
	
	parser.add_argument('--cloud', type=bool, default=True, required=False,
	                    help='choose location of dataset')
	
	# *********trigger info*********
	#  square  apple  watermark
	parser.add_argument('--trigger', type=str, default='square',
	                    help="Kind of trigger", choices=['square', 'pattern', 'watermark', 'apple'])
	
	# mnist 28*28  cifar10 32*32
	parser.add_argument('--triggerX', type=int, default='14', help="position of trigger x-aix")
	
	parser.add_argument('--triggerY', type=int, default='14', help="position of trigger y-aix")
	
	parser.add_argument('--verbose', default=True, action='store_true', help='verbose print')
	
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	
	parser.add_argument('--wrong_mal', type=int, default=0)
	
	parser.add_argument('--right_ben', type=int, default=0)
	
	parser.add_argument('--mal_score', type=float, default=0)
	
	parser.add_argument('--ben_score', type=float, default=0)
	
	parser.add_argument('--turn', type=int, default=0)
	
	parser.add_argument('--noise', type=float, default=0.001)
	
	parser.add_argument('--all_clients', default=False, action='store_true', help='aggregation over all clients')
	
	parser.add_argument('--debug', type=int, default=0, help="log debug info or not")
	
	parser.add_argument('--ablation_dataset', type=int, default=0, help="ablation experiment for dataset")
	
	parser.add_argument('--debug_fld', type=int, default=0, help="#1 save, #2 load")
	
	parser.add_argument('--decrease', type=float, default=0.3,
	                    help="proportion of dropped layers in robust experiments (used in mode11)")
	
	parser.add_argument('--increase', type=float, default=0.3,
	                    help="proportion of added layers in robust experiments (used in mode12)")
	
	parser.add_argument('--mode10_tau', type=float, default=0.95, help="threshold of mode 10")
	
	parser.add_argument('--cnn_scale', type=float, default=0.5, help="scale of cnn")
	
	parser.add_argument('--cifar_scale', type=float, default=1.0, help="scale of larger model")
	
	args = parser.parse_args()
	return args
