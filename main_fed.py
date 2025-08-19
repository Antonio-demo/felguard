##!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
#from random import random
import random
from models.test import test_img, test_label
from models.Fed import FedAvg
from models.Nets import ResNet18, vgg19_bn, vgg19, get_model, vgg11, CNN_MNIST, ResNet34, CNN_CIFAR, \
	preactresnet18,LoanNet,CNN_GTSRB
from models.resnet20 import resnet20
from models.MaliciousUpdate import LocalMaliciousUpdate
from models.Update import LocalUpdate
from utils.info import print_exp_details, write_info_to_accfile, get_base_info
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, mnist_noniid_dirichlet,\
	gtsrb_noniid_dirichlet
from utils.defense import fltrust, multi_krum, get_update, RLR, flame, get_update2, fld_distance, detection, detection1,\
	parameters_dict_to_vector_flt, lbfgs_torch, iid, sample_dirichlet_train_data, simple_median, FELGuardAvg,\
	felguard_distance, felguard_detection, FELGuardAvg_1,Repeated_Median_Shard,get_valid_models,\
	IRLS_aggregation_split_restricted, irls_aggregation_split_restricted,IRLS_aggregation_split_restricted_my,\
	trimmed_mean
from utils.LoanHelper import LoanHelper

from models.Attacker import attacker
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import os
import math
import pandas as pd
import openpyxl
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import gc

gc.collect()
torch.cuda.empty_cache()


def write_file(filename, accu_list, back_list, args, analyse=False):
	write_info_to_accfile(filename, args)
	f = open(filename, "a")
	f.write("main_task_accuracy=")
	f.write(str(accu_list))
	f.write('\n')
	f.write("backdoor_accuracy=")
	f.write(str(back_list))
	if args.defence == "krum":
		krum_file = filename + "_krum_dis"
		torch.save(args.krum_distance, krum_file)
	if analyse == True:
		need_length = len(accu_list) // 10
		acc = accu_list[-need_length:]
		back = back_list[-need_length:]
		best_acc = round(max(acc), 2)
		average_back = round(np.mean(back), 2)
		best_back = round(max(back), 2)
		f.write('\n')
		f.write('BBSR:')
		f.write(str(best_back))
		f.write('\n')
		f.write('ABSR:')
		f.write(str(average_back))
		f.write('\n')
		f.write('max acc:')
		f.write(str(best_acc))
		f.write('\n')
		f.close()
		return best_acc, average_back, best_back
	f.close()


def central_dataset_iid(dataset, dataset_size):
	all_idxs = [i for i in range(len(dataset))]
	central_dataset = set(np.random.choice(
		all_idxs, dataset_size, replace=False))
	return central_dataset


def test_mkdir(path):
	if not os.path.isdir(path):
		os.mkdir(path)


def loan_sample_by_state(loanHelper, num_users):
	keys = copy.deepcopy(loanHelper.state_keys)
	random.shuffle(keys)
	
	print("all addr_state :", keys)
	dict_users = {i: np.array([]) for i in range(num_users)}
	for i in range(num_users):
		idxs = loanHelper.dict_by_states[keys[i]]
		dict_users[i] = np.array(idxs)
		print('assigned {len(idxs)} data in addr_state {keys[i]} to user {i}')
	return dict_users


if __name__ == '__main__':
	# parse args
	args = args_parser()
	args.device = torch.device('cuda:{}'.format(
		args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
	test_mkdir('./' + args.save)
	print_exp_details(args)
	
	# load dataset and split users
	if args.dataset == 'mnist':
		trans_mnist = transforms.Compose(
			[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		dataset_train = datasets.MNIST(
			'./data/mnist/', train=True, download=True, transform=trans_mnist)
		dataset_test = datasets.MNIST(
			'./data/mnist/', train=False, download=True, transform=trans_mnist)
		
		# sample users
		if args.iid:
			dict_users = mnist_iid(dataset_train, args.num_users)
		else:
			#dict_users = mnist_noniid(dataset_train, args.num_users)
			dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users)
		
		# if args.iid:
		# 	dict_users = iid(dataset_train, args.num_users)
		# else:
		# 	dict_users = sample_dirichlet_train_data(dataset_train, args.num_users * (1 - args.malicious),
		#                                          args.num_users * args.malicious)
	
	elif args.dataset == 'fmnist':
		trans_mnist = transforms.Compose([transforms.ToTensor(),
		                                  transforms.Normalize(mean=[0.2860], std=[0.3530])])
		
		dataset_train = datasets.FashionMNIST(
			'./data/fmnist', train=True, download=True, transform=trans_mnist)
		
		dataset_test = datasets.FashionMNIST(
			'./data/fmnist', train=False, download=True, transform=trans_mnist)
		
		if args.iid:
			dict_users = iid(dataset_train, args.num_users)
		else:
			dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users)
			
		# sample users
		# if args.iid:
		# 	dict_users = np.load('./data/iid_fashion_mnist.npy', allow_pickle=True).item()
		# else:
		# 	dict_users = np.load('./data/non_iid_fashion_mnist.npy', allow_pickle=True).item()
	elif args.dataset == 'gtsrb':
		# 数据转换
		trans_gtsrb = transforms.Compose([
			transforms.Resize((32, 32)),  # 调整图像大小到 32x32
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669])  # GTSRB 数据集的标准化参数
		])
		
		# 加载训练集
		dataset_train = datasets.GTSRB(
			root='./data/gtsrb',  # 替换为实际存储路径
			split='train',
			download=True,  # 下载数据集
			transform=trans_gtsrb)
		
		# 加载测试集
		dataset_test = datasets.GTSRB(
			root='./data/gtsrb',  # 替换为实际存储路径
			split='test',
			download=True,  # 下载数据集
			transform=trans_gtsrb)
		
		# 数据划分
		if args.iid:
			dict_users = iid(dataset_train, args.num_users)
		else:
			#dict_users = sample_dirichlet_train_data(dataset_train, args.num_users, args.num_users * args.malicious)  # 使用非独立同分布划分
			dict_users = gtsrb_noniid_dirichlet(dataset_train, args.num_users)
			
	elif args.dataset == 'cifar':
		trans_cifar = transforms.Compose(
			[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		dataset_train = datasets.CIFAR10(
			'./data/cifar', train=True, download=True, transform=trans_cifar)
		
		dataset_test = datasets.CIFAR10(
			'./data/cifar', train=False, download=True, transform=trans_cifar)
		if args.iid:
			dict_users = np.load('./data/iid_cifar.npy', allow_pickle=True).item()
		else:
			dict_users = np.load('./data/non_iid_cifar.npy', allow_pickle=True).item()
	
	elif args.dataset == 'cifar100':
		
		trans_cifar100 = transforms.Compose([transforms.ToTensor(),
		                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		
		dataset_train = datasets.CIFAR100('./data/cifar100',
		                                  train=True, download=True, transform=trans_cifar100)
		
		dataset_test = datasets.CIFAR100('./data/cifar100',
		                                 train=False, download=True, transform=trans_cifar100)
		
		if args.iid:
			dict_users = iid(dataset_train, args.num_users)
		else:
			dict_users = sample_dirichlet_train_data(dataset_train, args.num_users,
			                                         args.num_users * args.malicious)
	
	elif args.dataset == 'loan':
		print('==> Preparing lending-club-loan-data..')
		filepath = './data/lending-club-loan-data/loan_processed.csv'
		loanHelper = LoanHelper(filepath)
		dataset_train = loanHelper.dataset_train
		dataset_test = loanHelper.dataset_test
		
		if args.iid:
			dict_users = iid(dataset_train, args.num_users)
		else:  # non-iid sample by states
			dict_users = loan_sample_by_state(loanHelper, args.num_users)
			
	else:
		exit('Error: unrecognized dataset')
	
	img_size = dataset_train[0][0].shape
	
	# build model
	# build model
	if args.model == "cnn" and args.dataset == 'mnist':
		net_glob = CNN_MNIST().to(args.device)
	
	elif args.model == "rlr_mnist" or args.model == "cnn" and args.dataset == 'fmnist':
		net_glob = get_model('fmnist').to(args.device)
	
	elif args.model == "vgg19" and args.dataset == 'cifar':
		net_glob = vgg19().to(args.device)
		
	elif args.model == "cnn" and args.dataset == 'gtsrb':
		net_glob = CNN_GTSRB(num_classes=43).to(args.device)
		
	elif args.model == "resnet" and args.dataset == "cifar":
		net_glob = ResNet34().to(args.device)
	
	elif args.model == "cnn" and args.dataset == "cifar":
		net_glob = CNN_CIFAR().to(args.device)
	
	elif args.model == "preactresnet18" and args.dataset == "cifar100":
		net_glob = preactresnet18().to(args.device)
	
	elif args.model == "loannet" and args.dataset == "loan":
		net_glob = LoanNet().to(args.device)
	
	else:
		exit('Error: unrecognized model')
	
	# if args.attack == 'baseline':
	# 	args.attack = 'badnet'
	# if args.defence == 'Fedavg':
	# 	args.defence = 'avg'
	# if args.model == 'cnn':
	# 	args.model = 'rlr_mnist'
	# net_glob.train()
	# if args.defence == 'fldetector':
	# 	args.defence = 'fld'
	
	# copy weights
	w_glob = net_glob.state_dict()
	
	# training
	loss_train = []
	cv_loss, cv_acc = [], []
	val_loss_pre, counter = 0, 0
	net_best = None
	best_loss = None
	
	if args.defence == 'fldetector':
		old_update_list = []
		weight_record = []
		update_record = []
		args.frac = 1
		malicious_score = torch.zeros((1, 100))
		
	if args.defence == 'felguard':
		old_update_list = []
		weight_record = []
		update_record = []
		args.frac = 1
		malicious_score = torch.zeros((1, 100))
	
	if math.isclose(args.malicious, 0):
		backdoor_begin_acc = 100
	else:
		backdoor_begin_acc = args.attack_begin  # overtake backdoor_begin_acc then attack
	
	central_dataset = central_dataset_iid(dataset_test, args.server_dataset)
	base_info = get_base_info(args)
	filename = './' + args.save + '/accuracy_file_{}.txt'.format(base_info)
	
	if args.init != 'None':
		param = torch.load(args.init)
		net_glob.load_state_dict(param)
		print("load init model")
	
	val_acc_list, net_list = [0.0001], []
	backdoor_acculist = [0]
	
	args.attack_layers = []
	
	if args.attack == "dba":
		args.dba_sign = 0
	if args.defence == "krum":
		args.krum_distance = []
	malicious_list = []
	for i in range(int(args.num_users * args.malicious)):
		malicious_list.append(i)
	
	if args.all_clients:
		print("Aggregation over all clients")
		w_locals = [w_glob for i in range(args.num_users)]
	
	results_df_1 = pd.DataFrame(columns=['Round', 'Train Average Loss', 'Train Average Acc', 'Train Total Time(s)'])
	
	results_df_2 = pd.DataFrame(columns=['Round', 'Validation Main Accuracy(%)', 'Attack Success Rate(%)',
	                                     'Validation Total Time(s)'])
	
	results_df_3 = pd.DataFrame(columns=['Final Training Accuracy(%)', 'Final Testing Accuracy(%)',
	                                     'Test Total Time(s)'])
	
	results_df_4 = pd.DataFrame(columns=['Malicious score'])
	
	tsne_results = pd.DataFrame(columns=['Label', 't-SNE 1', 't-SNE 2'])
	
	reweights = []
	
	print('start training!')
	for iter in range(args.epochs):
		start_train_time = time.time()
		
		loss_locals = []
		acc_locals = []
		
		if not args.all_clients:
			w_locals = []
			w_updates = []
		
		m = max(int(args.frac * args.num_users), 1)
		idxs_users = np.random.choice(range(args.num_users), m, replace=False)
		
		if args.defence == 'fldetector':
			idxs_users = np.arange(args.num_users)
			if iter == 350:
				args.lr *= 0.1
		elif args.defence == 'felguard':
			idxs_users = np.arange(args.num_users)
			if iter == 350:
				args.lr *= 0.1
		
		if backdoor_begin_acc < val_acc_list[-1]:
			backdoor_begin_acc = 0
			attack_number = int(args.malicious * m)
		else:
			attack_number = 0
		
		skip_number = 0
		mal_weight = []
		mal_loss = []
		args.attack_layers = []
		
		for num_turn, idx in enumerate(idxs_users):
			if attack_number > 0 and skip_number == 0:
				if args.defence == 'fldetector':
					args.old_update_list = old_update_list[0:int(args.malicious * m)]
					m_idx = idx
				elif args.defence == 'felguard':
					args.old_update_list = old_update_list[0:int(args.malicious * m)]
					m_idx = idx
				else:
					m_idx = None
				#print("skip number == 0")
				
				mal_weight, loss, args.attack_layers = attacker(malicious_list, attack_number, args.attack,
				                                                dataset_train, dataset_test, dict_users, net_glob, args,
				                                                idx=m_idx)
				
				attack_number -= 1
				if args.attack == 'adaptive':
					skip_number = attack_number
				if skip_number == 0:
					w = mal_weight[0]
				else:
					w = mal_weight[0]
			elif skip_number > 0:
				w = mal_weight[-skip_number]
				skip_number -= 1
				attack_number -= 1
			else:
				local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
				#w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device))
				w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
				
			if args.defence == 'fldetector':
				w_updates.append(get_update2(w, w_glob))  # ignore num_batches_tracked, running_mean, running_var
			elif args.defence == 'felguard':
				w_updates.append(get_update2(w, w_glob))
			else:
				w_updates.append(get_update(w, w_glob))
			
			if args.all_clients:
				w_locals[idx] = copy.deepcopy(w)
			else:
				w_locals.append(copy.deepcopy(w))
		
			#w_locals.append(copy.deepcopy(w))
			loss_locals.append(copy.deepcopy(loss))
		#acc_locals.append(copy.deepcopy(acc))
		
		if args.defence == 'avg':  # no defence
			w_glob = FedAvg(w_locals)
		elif args.defence == 'krum':  # single krum
			selected_client = multi_krum(w_updates, 1, args)
			# print(args.krum_distance)
			w_glob = w_locals[selected_client[0]]
			# w_glob = FedAvg([w_locals[i] for i in selected_clinet])
		elif args.defence == 'median':
			print("using simple median Estimator")
			w_glob = simple_median(w_locals)
		elif args.defence == 'repeated_median':
			print("using repeated median")
			w_locals, invalid_model_idx = get_valid_models(w_locals)
			w_glob = Repeated_Median_Shard(w_locals)
		
		elif args.defence == 'trimmed_mean':
			print("using trimmed mean Estimator")
			w_locals, invalid_model_idx = get_valid_models(w_locals)
			w_glob = trimmed_mean(w_locals, args.alpha_tri)
			
		elif args.defence == 'irls':
			print("using IRLS Estimator")  # residual-based re-weighting
			w_locals, invalid_model_idx = get_valid_models(w_locals)
			w_glob, reweight = IRLS_aggregation_split_restricted(w_locals, args.Lambda, args.thresh)
		
		elif args.defence == 'irls_reputation':
			print("using IRLS+Reputation Estimator")  # NDSS
			w_locals, invalid_model_idx = get_valid_models(w_locals)
			w_glob, reweight = irls_aggregation_split_restricted(w_locals,
			                                                     args.Lambda,
			                                                     args.thresh,
			                                                     args.reputation_active,
			                                                     args.reputation_effect,
			                                                     args.reputation_timestamp,
			                                                     args.kappa,
			                                                     args.W,
			                                                     args.eta,
			                                                     args.a,
			                                                     args.z
			                                                     )
		
		elif args.defence == 'rocba':
			# AGP+ADM+RMC
			print("using IRLS+Beta Reputation Estimator")  # Ours scheme
			w_locals, invalid_model_idx = get_valid_models(w_locals)
			w_glob, reweight = IRLS_aggregation_split_restricted_my(w_locals, args.loss_diff, args.reputation_active,
			                                                        args.reputation_effect,
			                                                        args.w_alpha, args.w_beta, args.alpha_i,
			                                                        args.beta_i,
			                                                        args.p_star,
			                                                        args.q_star, args.p, args.q, args.rho,
			                                                        args.Lambda, args.thresh)
			print(reweight)
			reweights.append(reweight)
		
		elif args.defence == 'multi_krum':
			selected_client = multi_krum(w_updates, args.k, args, multi_k=True)
			# print(selected_client)
			w_glob = FedAvg([w_locals[x] for x in selected_client])
		elif args.defence == 'RLR':
			w_glob = RLR(copy.deepcopy(net_glob), w_updates, args)
		elif args.defence == 'fltrust':
			local = LocalUpdate(args=args, dataset=dataset_test, idxs=central_dataset)
			fltrust_norm, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
			fltrust_norm = get_update(fltrust_norm, w_glob)
			w_glob = fltrust(w_updates, fltrust_norm, w_glob, args)
		elif args.defence == 'flame':
			w_glob = flame(w_locals, w_updates, w_glob, args, debug=args.debug)
		
		
		elif args.defence == 'fldetector':
			# ignore key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var'
			N = 5
			args.N = N
			weight = parameters_dict_to_vector_flt(w_glob)
			local_update_list = []
			for local in w_updates:
				local_update_list.append(-1 * parameters_dict_to_vector_flt(local).cpu())  # change to 1 dimension
			
			if iter > N + 1:
				hvp = lbfgs_torch(args, weight_record, update_record, weight - last_weight)
				
				attack_number = int(args.malicious * m)
				distance = fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp)
				distance = distance.view(1, -1)
				print('main.py line 320 distance:', distance)
				malicious_score = torch.cat((malicious_score, distance), dim=0)
				if malicious_score.shape[0] > N + 1:
					if detection1(np.sum(malicious_score[-N:].numpy(), axis=0)):
						
						label = detection(np.sum(malicious_score[-N:].numpy(), axis=0), int(args.malicious * m))
					else:
						label = np.ones(100)
					selected_client = []
					for client in range(100):
						if label[client] == 1:
							selected_client.append(client)
					new_w_glob = FedAvg([w_locals[client] for client in selected_client])
				else:
					new_w_glob = FedAvg(w_locals)  # avg
			else:
				hvp = None
				new_w_glob = FedAvg(w_locals)  # avg
			
			update = get_update2(w_glob, new_w_glob)  # w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
			update = parameters_dict_to_vector_flt(update)
			if iter > 0:
				weight_record.append(weight.cpu() - last_weight.cpu())
				update_record.append(update.cpu() - last_update.cpu())
			if iter > N:
				del weight_record[0]
				del update_record[0]
			
			last_weight = weight
			last_update = update
			old_update_list = local_update_list
			w_glob = new_w_glob
		
		elif args.defence == 'felguard':
			# This is the source code of our scheme, FELGuard
			# N = 5
			N = 250
			args.N = N
			weight = parameters_dict_to_vector_flt(w_glob)
			local_update_list = []
			
			for local in w_updates:
				local_update_list.append(-1 * parameters_dict_to_vector_flt(local).cpu())  # change to 1 dimension
			
			if iter > N + 1:
				hvp = lbfgs_torch(args, weight_record, update_record, weight - last_weight)
				
				attack_number = int(args.malicious * m)
				distance, angle_in_radians = felguard_distance(old_update_list, local_update_list, net_glob,
				                                               attack_number, hvp, args)
				
				distance = distance.view(1, -1)
				print('FELGuard distance:\n', distance)
				
				
				
				malicious_score = torch.cat((malicious_score, distance), dim=0)
				
				if malicious_score.shape[0] > N + 1:
					label, detection_acc, recall, fpr, fnr = felguard_detection(np.sum(malicious_score[-N:].numpy(),
					                                                                   axis=0), int(args.malicious * m))
					
					selected_client = []
					benign_indices = []
					
					for client in range(100):
						if label[client] == 1:
							selected_client.append(client)
							benign_indices.append(client)
						
						# selected_client.append(client)
						# benign_indices.append(client)#禁用聚类机制
					
					compute_malicious_score = np.sum(malicious_score[-N:].numpy(), axis=0)
					results_df_4 = results_df_4.append({'Malicious score': compute_malicious_score},
					                                   ignore_index=True)
					
					
					# 提取良性客户端对应的弧度
					benign_angles = angle_in_radians[benign_indices]
					
					# 计算良性客户端弧度的平均值
					mean_angle = torch.mean(benign_angles)
					# 计算非线性函数Gompertz function
					f_theta = 1 / (1 - torch.exp(- args.alpha * torch.exp((benign_angles - 1))))
					# 计算ψ_t^{(i)} = exp(f(θ)) / Σ exp(f(θ))
					exp_f_theta = torch.exp(f_theta)
					psi_values = exp_f_theta / torch.sum(exp_f_theta)
					
					# 应用 ψ 值进行联邦平均，使用 scale_model_weights 函数
					w_locals_trimmed = w_locals[:len(psi_values)]
					
					
					#new_w_glob = FELGuardAvg_1([w_locals[client] for client in benign_indices])  # FedAvg 函数需要修改为适应加权更新
				
					new_w_glob = FELGuardAvg_1(w_locals)#这是不加聚类的方法
				
				else:
					new_w_glob = FedAvg(w_locals)  # avg
			
			else:
				hvp = None
				new_w_glob = FedAvg(w_locals)  # avg
				
				update = get_update2(w_glob, new_w_glob)  # w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
				update = parameters_dict_to_vector_flt(update)
				
				if iter > 0:
					weight_record.append(weight.cpu() - last_weight.cpu())
					update_record.append(update.cpu() - last_update.cpu())
				if iter > N:
					del weight_record[0]
					del update_record[0]
				
				last_weight = weight
				last_update = update
				old_update_list = local_update_list
				w_glob = new_w_glob
		
		else:
			print("Wrong Defense Method")
			os._exit(0)
		
		# copy weight to net_glob
		net_glob.load_state_dict(w_glob)
		
		# print loss
		loss_avg = sum(loss_locals) / len(loss_locals)
		#acc_avg = sum(acc_locals) / len(acc_locals)
		#print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
		
		train_end_time = time.time()
		
		train_total_time = train_end_time - start_train_time
		
		print('\nRound {:3d}, Training Average Loss:{:.3f}, '
		      'Training Time: {:d}s'.format(iter + 1, loss_avg, int(train_total_time)))
		
		loss_train.append(loss_avg)
		
		# 将训练损失值和测试准确度结果添加到DataFrame中
		# 将训练损失值和测试准确度结果添加到DataFrame中
		results_df_1 = results_df_1.append({'Round': iter + 1,
		                                    'Train Average Loss': loss_avg,
		                                    'Train Total Time(s)': train_total_time, }, ignore_index=True)
		
		if iter % 1 == 0:
			start_valid_time = time.time()
			
			acc_test, _, back_acc = test_img(net_glob, dataset_test, args, test_backdoor=True)
			
			# print("Main accuracy: {:.2f}".format(acc_test))
			# print("Backdoor accuracy: {:.2f}".format(back_acc))
			val_acc_list.append(acc_test.item())
			
			backdoor_acculist.append(back_acc)
			write_file(filename, val_acc_list, backdoor_acculist, args)
			
			valid_end_time = time.time()
			valid_total_time = valid_end_time - start_valid_time
			
			print("Round {:3d}, Validation Main Accuracy: {:.2f}%, Backdoor Accuracy: {:.2f}%,"
			      " Validation Time: {:d}s".format
			      (iter + 1, acc_test, back_acc, int(valid_total_time)))
			
			results_df_2 = results_df_2.append({'Round': iter + 1,
			                                    'Validation Main Accuracy(%)': acc_test.item(),
			                                    'Attack Success Rate(%)': back_acc,
			                                    'Validation Total Time(s)': valid_total_time, }, ignore_index=True)
	
	best_acc, absr, bbsr = write_file(filename, val_acc_list, backdoor_acculist, args, True)
	
	
	# testing
	test_start_time = time.time()
	
	# testing
	net_glob.eval()
	acc_train, loss_train = test_img(net_glob, dataset_train, args)
	
	acc_test, loss_test = test_img(net_glob, dataset_test, args)
	test_end_time = time.time()
	
	test_total_time = test_end_time - test_start_time
	
	# # 获取真实标签和预测标签
	# test_loader = DataLoader(dataset=dataset_test, batch_size=args.bs, shuffle=False)
	# true_labels, pred_labels, test_features, flipped_labels = test_label(net_glob, test_loader, args)
	#
	# # 计算相关性系数
	# confusion_mtx = confusion_matrix(true_labels, pred_labels)
	# correlation_matrix = np.corrcoef(confusion_mtx)
	# # correlation_matrix = np.corrcoef(true_labels, pred_labels)
	#
	# print(f"相关性系数矩阵：{correlation_matrix},混淆矩阵：{confusion_mtx}")
	#
	# # 设置横坐标和纵坐标为数值 0 到 9，间隔为 1
	# labels = range(10)  # 假设标签范围为 0 到 9
	#
	# # 创建 DataFrame，横坐标为真实标签，纵坐标为预测标签
	# df_1 = pd.DataFrame(confusion_mtx,
	#                   index=labels,  # 纵坐标设置为 0-9
	#                   columns=labels)  # 横坐标设置为 0-9
	#
	# # 保存为 Excel 文件
	# excel_filename = './confusion_matrix.xlsx'
	# df_1.to_excel(excel_filename, index=True)
	# print(f"Excel 文件已保存到 {excel_filename}")
	#
	# # 创建 DataFrame，横坐标为真实标签，纵坐标为预测标签
	# df_2 = pd.DataFrame(correlation_matrix,
	#                   index=labels,  # 纵坐标设置为 0-9
	#                   columns=labels)  # 横坐标设置为 0-9
	#
	# # 保存为 Excel 文件
	# excel_filename = './correlation_matrix.xlsx'
	# df_2.to_excel(excel_filename, index=True)
	# print(f"Excel 文件已保存到 {excel_filename}")
	#
	# # 绘制相关性热力图
	# plt.figure(figsize=(10, 8))
	# sns.heatmap(correlation_matrix.T, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=range(10),
	#             yticklabels=range(10))
	# plt.title('Correlation Heatmap B etween True and Predicted Labels')
	# plt.xlabel('True Labels')  # 将x轴标签改为 True Labels
	# plt.ylabel('Predicted Labels')  # 将y轴标签改为 Predicted Labels
	# title = "Correlation Heatmap"
	# plt.title(title)
	# plt.savefig('./FLDetector_origin/FLDetector_origin/' +
	#             title + '.pdf', format='pdf', bbox_inches='tight')
	#
	# # 提取测试集的特征
	# tsne = TSNE(n_components=2, perplexity=40, learning_rate=500, n_iter=3000, random_state=42)
	# # 使用 PCA 降维到 50 维
	# pca = PCA(n_components=10)
	# test_features_2d = pca.fit_transform(test_features)
	#
	# # 绘制标签翻转攻击的样本
	# flipped_attack_idx = np.where((true_labels == 5) & (flipped_labels == 7))  # 标签 5 被翻转成标签 7 的索引
	# non_flipped_idx = np.where(~((true_labels == 5) & (flipped_labels == 7)))  # 未被翻转的样本
	#
	# plt.figure(figsize=(10, 8))
	# # 使用指定的颜色映射
	# palette = plt.get_cmap('tab10', 10)  # 使用 'tab10' 调色板
	# scatter = plt.scatter(test_features_2d[non_flipped_idx, 0], test_features_2d[non_flipped_idx, 1],
	#                       c=flipped_labels[non_flipped_idx], cmap='tab10', alpha=0.7, label="Normal Labels")
	#
	# # 单独绘制翻转样本
	# plt.scatter(test_features_2d[flipped_attack_idx, 0], test_features_2d[flipped_attack_idx, 1],
	#             c='black', s=150, label="Flipped: 5 -> 7", alpha=0.8)
	#
	# # plt.colorbar(scatter, ticks=range(11))  # 颜色条显示从 0 到 10 的标签
	# # plt.colorbar(scatter, ticks=range(10))
	# # plt.xlabel("t-SNE Component 1")
	# # plt.ylabel("t-SNE Component 2")
	# title_sne = "t-SNE visualization"
	# plt.title(title_sne)
	# plt.savefig('./FLDetector_origin/FLDetector_origin/' +
	#             title_sne + '.pdf', format='pdf', bbox_inches='tight')

	# 将 t-SNE 数据点逐行写入 DataFrame 中
	# for i in range(len(pred_labels)):
	# 	tsne_results = tsne_results.append({'Label': flipped_labels[i],
	#                                     't-SNE 1': test_features_2d[i, 0],
	#                                     't-SNE 2': test_features_2d [i, 1],}, ignore_index=True)

	# # 保存 t-SNE 数据到 Excel
	# tsne_results = pd.DataFrame({
	# 	'True Label': true_labels,
	# 	'Flipped Label': flipped_labels,
	# 	't-SNE 1': test_features_2d[:, 0],
	# 	't-SNE 2': test_features_2d[:, 1]
	# })
	#
	# excel_filename = './tSNE_with_flipped_labels.xlsx'
	#
	# tsne_results.to_excel(excel_filename, index=False)

	# print(f"t-SNE 数据已保存到：{excel_filename}")`
	
	
	print("Final Training Accuracy: {:.2f}%, Final Testing Accuracy: {:.2f}%, Test Time: {:d}s".format
	      (acc_train, acc_test, int(test_total_time)))
	
	results_df_3 = results_df_3.append({'Final Training Accuracy(%)': acc_train.item(),
	                                    'Final Testing Accuracy(%)': acc_test.item(),
	                                    'Test Total Time(s)': test_total_time, }, ignore_index=True)
	                                    
	# 将结果保存到 Excel文件
	with pd.ExcelWriter(
			f'{args.num_users}_{args.epochs}_{args.dataset}_{args.frac}_{args.model}_{args.defence}_{args.attack}_'
			f'_train_results.xlsx') as writer_1:
		results_df_1.to_excel(writer_1, index=False, sheet_name='Results')
	
	with pd.ExcelWriter(
			f'{args.num_users}_{args.epochs}_{args.dataset}_{args.frac}_{args.model}_{args.defence}_{args.attack}_'
			f'_validation_results.xlsx') as writer_2:
		results_df_2.to_excel(writer_2, index=False, sheet_name='Results')
	
	with pd.ExcelWriter(
			f'{args.num_users}_{args.epochs}_{args.dataset}_{args.frac}_{args.model}_{args.defence}_{args.attack}_'
			f'_test_results.xlsx') as writer_3:
		results_df_3.to_excel(writer_3, index=False, sheet_name='Results')
	
	with pd.ExcelWriter(
			f'{args.num_users}_{args.epochs}_{args.dataset}_{args.frac}_{args.model}_{args.defence}_{args.attack}_'
			f'_malicious_results.xlsx') as writer_4:
		results_df_4.to_excel(writer_4, index=False, sheet_name='Results')
	
	# print("Training accuracy: {:.2f}".format(acc_train))
	# print("Testing accuracy: {:.2f}".format(acc_test))
	
	#torch.save(net_glob.state_dict(), './' + args.save + '/model' + '.pth')



















