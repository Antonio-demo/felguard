#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import collections
import copy
import torch
from torch import nn
import time
import numpy as np
import sklearn.metrics.pairwise as smp
from functools import reduce
import math

eps = np.finfo(float).eps


def FedAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			try:
				w_avg[k] += w[i][k]
			except:
				print("Fed.py line17 type_as")
				w[i][k] = w[i][k].type_as(w_avg[k])
				w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg


def scale_model_weights(model_weights, scale_factor):
	# 对 OrderedDict 中的每个权重进行缩放
	scaled_weights = collections.OrderedDict()
	for key, value in model_weights.items():
		scaled_weights[key] = value * scale_factor
	return scaled_weights


def FELGuardAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			try:
				w_avg[k] += w[i][k]
			except:
				print("Fed.py line17 type_as")
				w[i][k] = w[i][k].type_as(w_avg[k])
				w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg


# simple median estimator
def simple_median(w):
	device = w[0][list(w[0].keys())[0]].device
	w_med = copy.deepcopy(w[0])
	cur_time = time.time()
	for k in w_med.keys():
		shape = w_med[k].shape
		if len(shape) == 0:
			continue
		total_num = reduce(lambda x, y: x * y, shape)
		y_list = torch.FloatTensor(len(w), total_num).to(device)
		for i in range(len(w)):
			y_list[i] = torch.reshape(w[i][k], (-1,))
		y = torch.t(y_list)
		median_result = median_opt(y)
		assert total_num == len(median_result)
		
		weight = torch.reshape(median_result, shape)
		w_med[k] = weight
	print('model aggregation "median" took {}s'.format(time.time() - cur_time))
	return w_med


def Repeated_Median_Shard(w):
	SHARD_SIZE = 100000
	cur_time = time.time()
	w_med = copy.deepcopy(w[0])
	device = w[0][list(w[0].keys())[0]].device
	
	for k in w_med.keys():
		shape = w_med[k].shape
		if len(shape) == 0:
			continue
		total_num = reduce(lambda x, y: x * y, shape)
		y_list = torch.FloatTensor(len(w), total_num).to(device)
		for i in range(len(w)):
			y_list[i] = torch.reshape(w[i][k], (-1,))
		y = torch.t(y_list)
		
		if total_num < SHARD_SIZE:
			slopes, intercepts = repeated_median(y)
			y = intercepts + slopes * (len(w) - 1) / 2.0
		else:
			y_result = torch.FloatTensor(total_num).to(device)
			assert total_num == y.shape[0]
			num_shards = int(math.ceil(total_num / SHARD_SIZE))
			for i in range(num_shards):
				y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
				slopes_shard, intercepts_shard = repeated_median(y_shard)
				y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
				y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
			y = y_result
		y = y.reshape(shape)
		w_med[k] = y
	
	print('repeated median aggregation took {}s'.format(time.time() - cur_time))
	return w_med


def trimmed_mean(w, trim_ratio):
	assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
	trim_num = int(trim_ratio * len(w))
	device = w[0][list(w[0].keys())[0]].device
	w_med = copy.deepcopy(w[0])
	cur_time = time.time()
	for k in w_med.keys():
		shape = w_med[k].shape
		if len(shape) == 0:
			continue
		total_num = reduce(lambda x, y: x * y, shape)
		y_list = torch.FloatTensor(len(w), total_num).to(device)
		for i in range(len(w)):
			y_list[i] = torch.reshape(w[i][k], (-1,))
		y = torch.t(y_list)
		y_sorted = y.sort()[0]
		result = y_sorted[:, trim_num:-trim_num]
		result = result.mean(dim=-1)
		assert total_num == len(result)
		
		weight = torch.reshape(result, shape)
		w_med[k] = weight
	print('model aggregation "trimmed mean" took {}s'.format(time.time() - cur_time))
	return w_med


def repeated_median(y):
	num_models = y.shape[1]
	total_num = y.shape[0]
	y = y.sort()[0]
	yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
	yyi = yyj.transpose(-1, -2)
	xx = torch.FloatTensor(range(num_models)).to(y.device)
	xxj = xx.repeat(total_num, num_models, 1)
	xxi = xxj.transpose(-1, -2) + eps
	
	diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
	diag = torch.diag(diag).repeat(total_num, 1, 1)
	
	dividor = xxi - xxj + diag
	slopes = (yyi - yyj) / dividor + diag
	slopes, _ = slopes.sort()
	slopes = median_opt(slopes[:, :, :-1])
	slopes = median_opt(slopes)
	
	# get intercepts (intercept of median)
	yy_median = median_opt(y)
	xx_median = [(num_models - 1) / 2.0] * total_num
	xx_median = torch.Tensor(xx_median).to(y.device)
	intercepts = yy_median - slopes * xx_median
	
	return slopes, intercepts


# 这是Residual-based re-weighting的代码
def IRLS_aggregation_split_restricted(w_locals, LAMBDA=2, thresh=0.1):
	SHARD_SIZE = 2000
	cur_time = time.time()
	w, invalid_model_idx = get_valid_models(w_locals)
	w_med = copy.deepcopy(w[0])
	# w_selected = [w[i] for i in random_select(len(w))]
	device = w[0][list(w[0].keys())[0]].device
	reweight_sum = torch.zeros(len(w)).to(device)
	
	for k in w_med.keys():
		shape = w_med[k].shape
		if len(shape) == 0:
			continue
		total_num = reduce(lambda x, y: x * y, shape)
		y_list = torch.FloatTensor(len(w), total_num).to(device)
		for i in range(len(w)):
			y_list[i] = torch.reshape(w[i][k], (-1,))
		transposed_y_list = torch.t(y_list)
		y_result = torch.zeros_like(transposed_y_list)
		assert total_num == transposed_y_list.shape[0]
		
		if total_num < SHARD_SIZE:
			reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
			reweight_sum += reweight.sum(dim=0)
			y_result = restricted_y
		else:
			num_shards = int(math.ceil(total_num / SHARD_SIZE))
			for i in range(num_shards):
				y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
				reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
				reweight_sum += reweight.sum(dim=0)
				y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y
		
		# put restricted y back to w
		y_result = torch.t(y_result)
		for i in range(len(w)):
			w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
	# print(reweight_sum)
	reweight_sum = reweight_sum / reweight_sum.max()
	reweight_sum = reweight_sum * reweight_sum
	w_med, reweight = weighted_average(w, reweight_sum)
	
	reweight = (reweight / reweight.max()).to(torch.device("cpu"))
	weights = torch.zeros(len(w_locals))
	i = 0
	for j in range(len(w_locals)):
		if j not in invalid_model_idx:
			weights[j] = reweight[i]
			i += 1
	
	print('model aggregation took {}s'.format(time.time() - cur_time))
	return w_med, weights


def weighted_average(w_list, weights):
	w_avg = copy.deepcopy(w_list[0])
	weights = weights / weights.sum()
	assert len(weights) == len(w_list)
	for k in w_avg.keys():
		w_avg[k] = 0
		for i in range(0, len(w_list)):
			w_avg[k] += w_list[i][k] * weights[i]
	# w_avg[k] = torch.div(w_avg[k], len(w_list))
	return w_avg, weights


def get_valid_models(w_locals):
	w, invalid_model_idx = [], []
	for i in range(len(w_locals)):
		if is_valid_model(w_locals[i]):
			w.append(w_locals[i])
		else:
			invalid_model_idx.append(i)
	return w, invalid_model_idx


def is_valid_model(w):
	if isinstance(w, list):
		w_keys = list(range(len(w)))
	else:
		w_keys = w.keys()
	for k in w_keys:
		params = w[k]
		if torch.isnan(params).any():
			return False
		if torch.isinf(params).any():
			return False
	return True


# 这是基于残差的攻击检测机制的方法
def reweight_algorithm_restricted(y, LAMBDA, thresh):
	num_models = y.shape[1]
	total_num = y.shape[0]
	slopes, intercepts = repeated_median(y)
	X_pure = y.sort()[1].sort()[1].type(torch.float)
	
	# calculate H matrix
	X_pure = X_pure.unsqueeze(2)
	X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
	X_X = torch.matmul(X.transpose(1, 2), X)
	X_X = torch.matmul(X, torch.inverse(X_X))
	H = torch.matmul(X_X, X.transpose(1, 2))
	diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
	processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
	K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)
	
	beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
	                  slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
	line_y = (beta * X).sum(dim=-1)
	residual = y - line_y
	M = median_opt(residual.abs().sort()[0][..., 1:])
	tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
	e = residual / tau.repeat(num_models, 1).transpose(0, 1)
	reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
	reweight[reweight != reweight] = 1
	reweight_std = reweight.std(dim=1)  # its standard deviation
	reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
	reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation
	
	restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
		torch.cuda.FloatTensor)
	return reweight_regulized, restricted_y


def foolsgold(grads):
	n_clients = grads.shape[0]
	cs = smp.cosine_similarity(grads) - np.eye(n_clients)
	maxcs = np.max(cs, axis=1)
	# pardoning
	for i in range(n_clients):
		for j in range(n_clients):
			if i == j:
				continue
			if maxcs[i] < maxcs[j]:
				cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
	wv = 1 - (np.max(cs, axis=1))
	wv[wv > 1] = 1
	wv[wv < 0] = 0
	
	# Rescale so that max value is wv
	wv = wv / np.max(wv)
	wv[(wv == 1)] = .99
	
	# Logit function
	wv = (np.log(wv / (1 - wv)) + 0.5)
	wv[(np.isinf(wv) + wv > 1)] = 1
	wv[(wv < 0)] = 0
	
	return wv


class FoolsGold(object):
	def __init__(self, args):
		self.memory = None
		self.wv_history = []
		self.args = args
	
	def aggregate_gradients(self, client_grads):
		cur_time = time.time()
		num_clients = len(client_grads)
		grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
		if self.memory is None:
			self.memory = np.zeros((num_clients, grad_len))
		
		grads = np.zeros((num_clients, grad_len))
		for i in range(len(client_grads)):
			grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
		
		if self.args.use_memory:
			self.memory += grads
			wv = foolsgold(self.memory)  # Use FG
		else:
			wv = foolsgold(grads)  # Use FG
		print(wv)
		self.wv_history.append(wv)
		
		agg_grads = []
		# Iterate through each layer
		for i in range(len(client_grads[0])):
			assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
				len(wv), len(client_grads))
			temp = wv[0] * client_grads[0][i].cpu().clone()
			# Aggregate gradients for a layer
			for c, client_grad in enumerate(client_grads):
				if c == 0:
					continue
				temp += wv[c] * client_grad[i].cpu()
			temp = temp / len(client_grads)
			agg_grads.append(temp)
		print('model aggregation took {}s'.format(time.time() - cur_time))
		return agg_grads


def median_opt(input):
	shape = input.shape
	input = input.sort()[0]
	if shape[-1] % 2 != 0:
		output = input[..., int((shape[-1] - 1) / 2)]
	else:
		output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
	return output
