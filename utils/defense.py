# -*- coding = utf-8 -*-
import math

import numpy as np
import torch
import copy
import random
import time
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
from collections import Counter
from functools import reduce

from sklearn.metrics import roc_auc_score


# from mxnet import nd
# import mxnet as mx

def cos(a, b):
	# res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
	res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
	      (np.linalg.norm(b) + 1e-9)
	'''relu'''
	if res < 0:
		res = 0
	return res


def FELGuardAvg(w, psi_values):
	"""
	加权的联邦平均算法，将每个客户端的模型更新按照 psi_values 的权重进行加权平均。

	参数:
	- w: 客户端的模型更新列表，w[i] 表示第 i 个客户端的更新
	- psi_values: 每个客户端的权重列表，长度与 w 一致

	返回:
	- w_avg: 加权后的全局模型更新
	"""
	# 检查 psi_values 和 w 长度是否一致
	if len(w) != len(psi_values):
		raise ValueError(f"w 和 psi_values 的长度不匹配：w 有 {len(w)} 个客户端，而 psi_values 有 {len(psi_values)} 个权重")
	
	# 初始化 w_avg 为第一个客户端的权重
	w_avg = copy.deepcopy(w[0])
	
	# 对每个权重参数进行加权平均
	for k in w_avg.keys():
		w_avg[k] = w_avg[k] * psi_values[0]  # 初始化为第一个客户端的加权值
		
		# 遍历其余客户端，并进行加权累加
		for i in range(1, len(w)):  # 确保不会超出索引范围
			try:
				w_avg[k] += w[i][k] * psi_values[i]  # 按权重加权累加
			except IndexError as e:
				print(f"索引错误：w[{i}] 或 psi_values[{i}] 越界 - {str(e)}")
				raise e  # 抛出异常，方便调试
			except KeyError as e:
				print(f"Key 错误：{str(e)}")
				raise e
			except Exception as e:
				# 处理数据类型不匹配的情况
				print("FELGuardAvg: type mismatch, adjusting type")
				w[i][k] = w[i][k].type_as(w_avg[k])
				w_avg[k] += w[i][k] * psi_values[i]
	
	return w_avg  # 返回加权平均后的模型


def FELGuardAvg_1(w):
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



def felguard_distance(old_update_list, local_update_list, net_glob, attack_number, hvp, args):
	# 计算预测的模型更新和当前的模型更新的距离
	pred_update = []
	distance = []
	alpha = args.alpha
	
	for i in range(len(old_update_list)):
		pred_update.append((old_update_list[i] + hvp).view(-1))
	
	pred_update = torch.stack(pred_update)
	local_update_list = torch.stack(local_update_list)
	
	# print('defence line 211 pred_update.shape:', pred_update.shape)
	euclidean_distance = torch.norm((pred_update - local_update_list), dim=1)
	# euclidean_distance = distance / torch.sum(distance)
	
	# 计算余弦距离(1-余弦相似度)
	cos_similarity = F.cosine_similarity(pred_update, local_update_list, dim=1)
	angle_in_radians = torch.acos(cos_similarity)
	
	# 限制弧度在 0 到 π/2 之间
	angle_in_radians = torch.clamp(angle_in_radians, 0, math.pi / 2)
	
	cosine_distance = 1 - cos_similarity
	
	# 计算加权组合距离, alpha * 欧氏距离 + (1 - alpha) * 余弦距离
	combined_distance = alpha * euclidean_distance + (1 - alpha) * cosine_distance
	# 归一化
	combined_distance = combined_distance / torch.sum(combined_distance)
	
	return combined_distance, angle_in_radians


def felguard_detection(score, nobyz):
	# FELGuard的检测方法
	# 使用 HDBSCAN 进行聚类
	clusterer = hdbscan.HDBSCAN(min_cluster_size=2)  # 设置最小聚类规模
	labels = clusterer.fit_predict(score.reshape(-1, 1))  # 聚类标签，-1 表示未分类数据
	label_pred_1 = clusterer.labels_
	
	if np.mean(score[label_pred_1 == 0]) < np.mean(score[label_pred_1 == 1]):
		# 0 is the label of malicious clients, 1 is the label of benign clients
		label_pred_1 = 1 - label_pred_1
	real_label = np.ones(100)
	real_label[:nobyz] = 0
	
	detection_acc = len(label_pred_1[label_pred_1 == real_label]) / 100
	recall = 1 - np.sum(label_pred_1[:nobyz]) / nobyz
	
	# Calculate precision
	# true_positives = np.sum((label_pred_1 == 1) & (real_label == 1))
	# false_positives = np.sum((label_pred_1 == 1) & (real_label == 0))
	#
	# precision = true_positives / (true_positives + false_positives)
	
	fpr = 1 - np.sum(label_pred_1[nobyz:]) / (100 - nobyz)
	fnr = np.sum(label_pred_1[:nobyz]) / nobyz
	
	print("Detection Acc: {:.2f}%; Recall: {:.4f}; FPR: {:.4f}; FNR: {:.4f};".format
	      (detection_acc * 100., recall, fpr, fnr))
	
	# 统计每个聚类的大小
	unique_labels, counts = np.unique(labels, return_counts=True)
	
	# 将较大的聚类视为良性客户端，较小的为恶意客户端
	benign_label = unique_labels[np.argmax(counts)]  # 良性客户端的聚类标签
	malicious_label = unique_labels[np.argmin(counts)]  # 恶意客户端的聚类标签
	
	# 标记每个客户端的标签，良性为1，恶意为0
	label_pred = np.ones(len(labels))
	label_pred[labels == malicious_label] = 0  # 小的聚类是恶意客户端
	
	# 获取良性客户端的标签
	benign_clients = np.where(label_pred == 1)[0]
	
	# 对良性客户端进行投票，统计每个标签的频率
	votes = Counter(labels[benign_clients])  # 统计每个标签的投票次数
	most_common_label, _ = votes.most_common(1)[0]  # 频率最高的标签
	
	# 返回投票结果，恶意客户端为0，良性客户端为1
	final_labels = np.zeros(100)
	final_labels[benign_clients] = 1  # 默认投票结果为良性客户端标签1
	
	return final_labels, detection_acc * 100., recall, fpr * 100., fnr * 100.

eps = np.finfo(float).eps

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

def get_valid_models(w_locals):
	w, invalid_model_idx = [], []
	for i in range(len(w_locals)):
		if is_valid_model(w_locals[i]):
			w.append(w_locals[i])
		else:
			invalid_model_idx.append(i)
	return w, invalid_model_idx

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


# 这是NDSS那篇论文的代码部分
def irls_aggregation_split_restricted(w_locals, LAMBDA, thresh, reputation_active, reputation_effect,
                                      reputation_timestamp, kappa, W, eta, a, z):
	"""
			Function to aggregate IRLS over Reputation Model
			:param w_locals:
			:param LAMBDA:
			:param thresh: is threshold to select to check if we use the update.
			:param kappa: the objective function value to fill out in the model when we consider R and S.
			:param eta (Î·): eta + k = 1, so eta = 1-k
			:param W: weight of the reputation we assigned initially, W=2 assigned in paper by default.
			:param a: a priori probability in the absence of committed belief mass. If we increase a, we scale up reputation.
			:param z: interaction freshness z (0,1). If we increase z we scale up our reputation model.
			:param reputation_active: parameter to control reputation model on/off.
			:param reputation_effect: parameter to store history of reputation
			:return: weights median and weights (w_med, weights)
	"""
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
	
	# Printing the reweight_sum before any reputation calculation
	# print("Printing the reweight_sum before any reputation calculation: " + str(reweight_sum))
	
	# Reputation effects code start
	print(str(reputation_active))
	if reputation_active:
		print("Reputation Active")  # 这里计算信誉值
		reweight_sum = reputation_model_reweighting(reweight_sum, reputation_effect, reputation_timestamp,
		                                            kappa, W, a, z)
	# Printing the reweight_sum before after reputation calculation
	# print("Printing the reweight_sum after any reputation calculation: " + str(reweight_sum))
	else:
		print("Reputation Inactive")
	
	w_med, reweight = weighted_average(w, reweight_sum)
	
	reweight = (reweight / reweight.max()).to(torch.device("cpu"))
	weights = torch.zeros(len(w_locals))
	i = 0
	for j in range(len(w_locals)):
		if j not in invalid_model_idx:
			weights[j] = reweight[i]
			i += 1
	print('model aggregation took {}s'.format(time.time() - cur_time))
	#  print('model reputation_effect {}'.format(reputation_effect))
	return w_med, weights


def reputation_model_reweighting(reweight_sum, reputation_effect, reputation_timestamp, kappa, W, a, z):
	"""
	Defined function for our reputation model at IMDEA NETWORKS
	BY @tianyuechu and @algarecu
	这里计算信誉值
	:return: reweighted sum
	"""
	for i in range(len(reweight_sum)):
		if reweight_sum[i] >= 0.6:
			belief = kappa / (kappa + W)
			uncertainty = W / kappa + W
		else:
			belief = 0
			uncertainty = W / (1 - kappa + W)
		
		if reputation_effect is None:
			reputation_effect = []
		else:
			reputation_effect = [0] * len(reweight_sum)
		
		reputation_effect.append(belief + a * uncertainty)
		
		# Restart at window size the reputation_timestamp = 0
		theta = [0] * len(reputation_effect)
		
		for j in range(len(reputation_effect)):
			# time decay function,即时间延迟函数
			# theta[j]=z^len(reputation_effect[i])-j-1
			theta[j] = pow(z, len(reputation_effect) - j - 1)
			
			reputation_timestamp += theta[j] * reputation_effect[j]
		# reputation_timestamp代表方案中计算得到的信誉值
		reputation_timestamp = reputation_timestamp / sum(theta)
		
		reweight_sum[i] = reputation_timestamp * reweight_sum[i]
	
	return reweight_sum


def IRLS_aggregation_split_restricted_my(w_locals, loss_diff, reputation_active, reputation_effect,
                                         reputation_timestamp, w_alpha, w_beta, alpha_i, beta_i, p_star, q_star, p, q,
                                         rho,
                                         LAMBDA=2, thresh=0.1):
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
	
	# Printing the reweight_sum before any reputation calculation
	# print("Printing the reweight_sum before any reputation calculation: " + str(reweight_sum))
	
	# Reputation effects code start
	print(str(reputation_active))
	if reputation_active:
		print("Reputation Active")  # 这里计算信誉值
		reweight_sum = reputation_model_reweighting_my(reweight_sum, loss_diff, reputation_effect,
		                                               reputation_timestamp,
		                                               w_alpha, p, w_beta, alpha_i, beta_i, p_star, q_star, rho, q)
	# Printing the reweight_sum before after reputation calculation
	# print("Printing the reweight_sum after any reputation calculation: " + str(reweight_sum))
	else:
		print("Reputation Inactive")
	
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

def u(x, rho):
	return np.tanh(rho * x)

def reputation_model_reweighting_my(reweight_sum, loss_diff, reputation_effect, reputation_timestamp, w_alpha,
                                    w_beta, alpha_i,
                                    beta_i, p_star, q_star, rho, p, q):
	"""
	Defined function for our reputation model at IMDEA NETWORKS
	BY @tianyuechu and @algarecu
	这里计算信誉值
	:return: reweighted sum
	"""
	if reputation_effect is None:
		reputation_effect = []
	else:
		reputation_effect = [0] * len(reweight_sum)
	
	for i in range(len(reweight_sum)):
		if reweight_sum[i] >= 0.6 and loss_diff > 0:
			# if loss_diff > 0:
			alpha_i_new = w_alpha * alpha_i + p_star * u(loss_diff, rho)
			beta_i_new = beta_i
			V_i = (p + alpha_i_new + 1) / (alpha_i_new + beta_i_new + p + q + 2)
		else:
			alpha_i_new = alpha_i
			beta_i_new = w_beta * beta_i - q_star * u(loss_diff, rho)
			V_i = (p + alpha_i_new + 1) / (alpha_i_new + beta_i_new + p + q + 2)
		
		if reputation_effect is None:
			reputation_effect = []
		else:
			reputation_effect = [0] * len(reweight_sum)
		
		reputation_effect.append(V_i)
		
		# Restart at window size the reputation_timestamp = 0
		theta = [0] * len(reputation_effect)
		
		for j in range(len(reputation_effect)):
			# time decay function,即时间延迟函数
			# theta[j]=z^len(reputation_effect[i])-j-1
			# theta[j] = pow(z, len(reputation_effect[i]) - j - 1)
			TD_1, TD_2, TD_3 = 0.3, 0.3, 0.4
			varepsilon, d_0, n = 1, 5, 3
			Theta = len(reputation_effect)
			theta[j] = TD_1 / Theta + TD_2 * (-0.125 * Theta + varepsilon) + TD_3 / (1 + pow(Theta, n) / pow(d_0, n))
			
			reputation_timestamp += theta[j] * reputation_effect[j]
		# reputation_timestamp代表方案中计算得到的信誉值
		reputation_timestamp = reputation_timestamp / sum(theta)
		
		reweight_sum[i] = reputation_timestamp * reweight_sum[i]
	
	return reweight_sum

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

def median_opt(input):
	shape = input.shape
	input = input.sort()[0]
	if shape[-1] % 2 != 0:
		output = input[..., int((shape[-1] - 1) / 2)]
	else:
		output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
	return output


def fltrust(params, central_param, global_parameters, args):
	FLTrustTotalScore = 0
	score_list = []
	central_param_v = parameters_dict_to_vector_flt(central_param)
	central_norm = torch.norm(central_param_v)
	cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
	sum_parameters = None
	
	for local_parameters in params:
		local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
		# 计算cos相似度得分和向量长度裁剪值
		client_cos = cos(central_param_v, local_parameters_v)
		client_cos = max(client_cos.item(), 0)
		client_clipped_value = central_norm / torch.norm(local_parameters_v)
		score_list.append(client_cos)
		FLTrustTotalScore += client_cos
		if sum_parameters is None:
			sum_parameters = {}
			for key, var in local_parameters.items():
				# 乘得分 再乘裁剪值
				sum_parameters[key] = client_cos * \
				                      client_clipped_value * var.clone()
		else:
			for var in sum_parameters:
				sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
					var]
	if FLTrustTotalScore == 0:
		# print(score_list)
		return global_parameters
	for var in global_parameters:
		# 除以所以客户端的信任得分总和
		temp = (sum_parameters[var] / FLTrustTotalScore)
		if global_parameters[var].type() != temp.type():
			temp = temp.type(global_parameters[var].type())
		if var.split('.')[-1] == 'num_batches_tracked':
			global_parameters[var] = params[0][var]
		else:
			global_parameters[var] += temp * args.server_lr
	# print(score_list)
	return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
	vec = []
	for key, param in net_dict.items():
		# print(key, torch.max(param))
		if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[
			-1] == 'running_var':
			continue
		vec.append(param.view(-1))
	return torch.cat(vec)


def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
	vec = []
	for key, param in net_dict.items():
		# print(key, torch.max(param))
		if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[
			-1] == 'running_var':
			continue
		vec.append(param.cpu().view(-1))
	return torch.cat(vec)


def no_defence_balance(params, global_parameters):
	total_num = len(params)
	sum_parameters = None
	for i in range(total_num):
		if sum_parameters is None:
			sum_parameters = {}
			for key, var in params[i].items():
				sum_parameters[key] = var.clone()
		else:
			for var in sum_parameters:
				sum_parameters[var] = sum_parameters[var] + params[i][var]
	for var in global_parameters:
		if var.split('.')[-1] == 'num_batches_tracked':
			global_parameters[var] = params[0][var]
			continue
		global_parameters[var] += (sum_parameters[var] / total_num)
	
	return global_parameters


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

def multi_krum(gradients, n_attackers, args, multi_k=False):
	grads = flatten_grads(gradients)
	
	candidates = []
	candidate_indices = []
	remaining_updates = torch.from_numpy(grads)
	all_indices = np.arange(len(grads))
	
	while len(remaining_updates) > 2 * n_attackers + 2:
		torch.cuda.empty_cache()
		distances = []
		scores = None
		for update in remaining_updates:
			distance = []
			for update_ in remaining_updates:
				distance.append(torch.norm((update - update_)) ** 2)
			distance = torch.Tensor(distance).float()
			distances = distance[None, :] if not len(
				distances) else torch.cat((distances, distance[None, :]), 0)
		
		distances = torch.sort(distances, dim=1)[0]
		scores = torch.sum(
			distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
		# print(scores)
		# args.krum_distance.append(scores)
		indices = torch.argsort(scores)[:len(
			remaining_updates) - 2 - n_attackers]
		
		candidate_indices.append(all_indices[indices[0].cpu().numpy()])
		all_indices = np.delete(all_indices, indices[0].cpu().numpy())
		candidates = remaining_updates[indices[0]][None, :] if not len(
			candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
		remaining_updates = torch.cat(
			(remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
		if not multi_k:
			break
	
	# aggregate = torch.mean(candidates, dim=0)
	
	# return aggregate, np.array(candidate_indices)
	num_clients = max(int(args.frac * args.num_users), 1)
	num_malicious_clients = int(args.malicious * num_clients)
	num_benign_clients = num_clients - num_malicious_clients
	args.turn += 1
	for selected_client in candidate_indices:
		if selected_client < num_malicious_clients:
			args.wrong_mal += 1
	
	# print(candidate_indices)
	
	# print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))
	
	for i in range(len(scores)):
		if i < num_malicious_clients:
			args.mal_score += scores[i]
		else:
			args.ben_score += scores[i]
	
	return np.array(candidate_indices)


def flatten_grads(gradients):
	param_order = gradients[0].keys()
	
	flat_epochs = []
	
	for n_user in range(len(gradients)):
		user_arr = []
		grads = gradients[n_user]
		for param in param_order:
			try:
				user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
			except:
				user_arr.extend(
					[grads[param].cpu().numpy().flatten().tolist()])
		flat_epochs.append(user_arr)
	
	flat_epochs = np.array(flat_epochs)
	
	return flat_epochs


def get_update(update, model):
	'''get the update weight'''
	update2 = {}
	for key, var in update.items():
		update2[key] = update[key] - model[key]
	return update2


def get_update2(update, model):
	'''get the update weight'''
	update2 = {}
	for key, var in update.items():
		if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[
			-1] == 'running_var':
			continue
		update2[key] = update[key] - model[key]
	return update2


def iid(dataset, num_users):
	"""
	Sample I.I.D. client data from CIFAR10 dataset
	:param dataset:
	:param num_users:
	:return: dict of image index
	"""
	num_items = int(len(dataset) / num_users)
	dict_users, all_idxs = {}, [i for i in range(len(dataset))]
	for i in range(num_users):
		dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
		all_idxs = list(set(all_idxs) - dict_users[i])
		print('assigned {} data to user {}'.format(len(dict_users[i]), i))
	return dict_users


def sample_dirichlet_train_data(dataset, num_users, num_attackers, alpha=0.5):
	classes = {}
	for idx, x in enumerate(dataset):
		_, label = x
		# label=label.item() # for gpu
		if label in classes:
			classes[label].append(idx)
		else:
			classes[label] = [idx]
	num_classes = len(classes.keys())
	class_size = len(classes[0])
	num_participants = num_users + num_attackers
	print(f"num_users + num_attackers:{num_users + num_attackers}")#130
	dict_users = {i: np.array([]) for i in range(num_users + num_attackers)}
	
	for n in range(num_classes):
		random.shuffle(classes[n])
		sampled_probabilities = class_size * np.random.dirichlet(np.array(num_participants * [alpha]))
		for user in range(num_participants):
			num_imgs = int(round(sampled_probabilities[user]))
			sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
			dict_users[user] = np.concatenate((dict_users[user], np.array(sampled_list)), axis=0)
			classes[n] = classes[n][min(len(classes[n]), num_imgs):]
	
	# shuffle data
	for user in range(num_participants):
		random.shuffle(dict_users[user])
	return dict_users


def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
	# 计算预测的模型更新和当前的模型更新的距离
	pred_update = []
	distance = []
	for i in range(len(old_update_list)):
		pred_update.append((old_update_list[i] + hvp).view(-1))
	
	pred_update = torch.stack(pred_update)
	local_update_list = torch.stack(local_update_list)
	# old_update_list = torch.stack(old_update_list)
	
	# distance = torch.norm((old_update_list - local_update_list), dim=1)
	# print('defense line219 distance(old_update_list - local_update_list):',distance)
	# auc1 = roc_auc_score(pred_update.numpy(), distance)
	# distance = torch.norm((pred_update - local_update_list), dim=1).numpy()
	# auc2 = roc_auc_score(pred_update.numpy(), distance)
	# print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))
	
	# print('defence line 211 pred_update.shape:', pred_update.shape)
	distance = torch.norm((pred_update - local_update_list), dim=1)
	distance = distance / torch.sum(distance)
	return distance


def felguard_distance(old_update_list, local_update_list, net_glob, attack_number, hvp, args):
	# 计算预测的模型更新和当前的模型更新的距离
	pred_update = []
	distance = []
	alpha = args.alpha
	
	for i in range(len(old_update_list)):
		pred_update.append((old_update_list[i] + hvp).view(-1))
	
	pred_update = torch.stack(pred_update)
	local_update_list = torch.stack(local_update_list)
	
	# print('defence line 211 pred_update.shape:', pred_update.shape)
	euclidean_distance = torch.norm((pred_update - local_update_list), dim=1)
	# euclidean_distance = distance / torch.sum(distance)
	
	# 计算余弦距离(1-余弦相似度)
	cos_similarity = F.cosine_similarity(pred_update, local_update_list, dim=1)
	angle_in_radians = torch.acos(cos_similarity)
	
	cosine_distance = 1 - cos_similarity
	
	#angle_in_radians = torch.clamp(angle_in_radians, -math.pi / 2, math.pi / 2)
	
	# 计算加权组合距离, alpha * 欧氏距离 + (1 - alpha) * 余弦距离
	combined_distance = alpha * euclidean_distance + (1 - alpha) * cosine_distance
	# 归一化
	combined_distance = combined_distance / torch.sum(combined_distance)
	
	return combined_distance, angle_in_radians


# def felguard_detection(score, nobyz):
# 	#FELGuard的检测方法
# 	#首先,使用HDBSCAN聚类
# 	cluster = hdbscan.HDBSCAN(min_cluster_size=2)
# 	cluster.fit(score.reshape(-1,1))
# 	label_pred = cluster.labels_
#
# 	#统计聚类的大小
# 	cluster_sizes = np.bincount(label_pred[label_pred >= 0]) #排除噪声标签-1
# 	sorted_clusters = np.argsort(cluster_sizes)[::-1] #按大小排序
#
# 	#在聚类假设中,最大的聚类为良性,最小的聚类为恶意
# 	if len(sorted_clusters) >= 2:
# 		benign_cluster = sorted_clusters[0]
# 		malicious_cluster = sorted_clusters[1]
# 	else:
# 		benign_cluster = sorted_clusters[0]
# 		malicious_cluster = -1 #没有足够的聚类
#
# 	#标记所有良性和恶意客户端
# 	real_label = np.ones(100)
# 	real_label[label_pred == malicious_cluster] = 0 #小的聚类为恶意
#
# 	#在良性聚类中进行投票
# 	benign_indices = np.where(label_pred == benign_cluster)[0]
# 	benign_vote = np.ones_like(real_label) #初始化全为良性标签
# 	if len(benign_indices) > 0:
# 		for i in benign_indices:
# 			#模拟每个客户端投票
# 			benign_vote[i] = 1 #良性投票标记为1
# 		#进行投票,使用HDBSCAN算法提取频率最高的更新
# 		highest_vote_update = cluster.outlier_scores_[benign_indices].argmax()


def felguard_detection(score, nobyz):
	# FELGuard的检测方法
	# 使用 HDBSCAN 进行聚类
	clusterer = hdbscan.HDBSCAN(min_cluster_size=2)  # 设置最小聚类规模
	labels = clusterer.fit_predict(score.reshape(-1, 1))  # 聚类标签，-1 表示未分类数据
	label_pred_1 = clusterer.labels_
	
	# if np.mean(score[label_pred_1 == 0]) < np.mean(score[label_pred_1 == 1]):
	# 	# 0 is the label of malicious clients, 1 is the label of benign clients
	# 	label_pred_1 = 1 - label_pred_1
	
	real_label = np.ones(100)
	real_label[:nobyz] = 0
	
	detection_acc = len(label_pred_1[label_pred_1 == real_label]) / 100
	recall = 1 - np.sum(label_pred_1[:nobyz]) / nobyz
	
	# Calculate precision
	# true_positives = np.sum((label_pred_1 == 1) & (real_label == 1))
	# false_positives = np.sum((label_pred_1 == 1) & (real_label == 0))
	#
	# precision = true_positives / (true_positives + false_positives)
	
	fpr = 1 - np.sum(label_pred_1[nobyz:]) / (100 - nobyz)
	fnr = np.sum(label_pred_1[:nobyz]) / nobyz
	
	print("Detection Acc: {:.2f}%; Recall: {:.4f}; FPR: {:.4f}; FNR: {:.4f};".format
	      (detection_acc * 100., recall, fpr, fnr))
	
	# 统计每个聚类的大小
	unique_labels, counts = np.unique(labels, return_counts=True)
	
	# 将较大的聚类视为良性客户端，较小的为恶意客户端
	benign_label = unique_labels[np.argmax(counts)]  # 良性客户端的聚类标签
	malicious_label = unique_labels[np.argmin(counts)]  # 恶意客户端的聚类标签
	
	# 标记每个客户端的标签，良性为1，恶意为0
	label_pred = np.ones(len(labels))
	label_pred[labels == malicious_label] = 0  # 小的聚类是恶意客户端
	
	# 获取良性客户端的标签
	benign_clients = np.where(label_pred == 1)[0]
	
	# 对良性客户端进行投票，统计每个标签的频率
	votes = Counter(labels[benign_clients])  # 统计每个标签的投票次数
	most_common_label, _ = votes.most_common(1)[0]  # 频率最高的标签
	
	# 返回投票结果，恶意客户端为0，良性客户端为1
	final_labels = np.zeros(100)
	final_labels[benign_clients] = 1  # 默认投票结果为良性客户端标签1
	
	return final_labels, detection_acc * 100., recall, fpr * 100., fnr * 100.


def detection(score, nobyz):
	estimator = KMeans(n_clusters=2)
	estimator.fit(score.reshape(-1, 1))
	label_pred = estimator.labels_
	
	if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
		# 0 is the label of malicious clients, 1 is the label of benign clients
		label_pred = 1 - label_pred
	real_label = np.ones(100)
	real_label[:nobyz] = 0
	
	acc = len(label_pred[label_pred == real_label]) / 100
	recall = 1 - np.sum(label_pred[:nobyz]) / nobyz
	fpr = 1 - np.sum(label_pred[nobyz:]) / (100 - nobyz)
	fnr = np.sum(label_pred[:nobyz]) / nobyz
	# print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
	# print(silhouette_score(score.reshape(-1, 1), label_pred))
	# print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
	return label_pred


def detection1(score):
	nrefs = 10
	ks = range(1, 8)
	gaps = np.zeros(len(ks))
	gapDiff = np.zeros(len(ks) - 1)
	sdk = np.zeros(len(ks))
	min = np.min(score)
	max = np.max(score)
	
	score = (score - min) / (max - min)
	
	for i, k in enumerate(ks):
		estimator = KMeans(n_clusters=k)
		estimator.fit(score.reshape(-1, 1))
		label_pred = estimator.labels_
		center = estimator.cluster_centers_
		Wk = np.sum([np.square(score[m] - center[label_pred[m]]) for m in range(len(score))])
		WkRef = np.zeros(nrefs)
		for j in range(nrefs):
			rand = np.random.uniform(0, 1, len(score))
			estimator = KMeans(n_clusters=k)
			estimator.fit(rand.reshape(-1, 1))
			label_pred = estimator.labels_
			center = estimator.cluster_centers_
			WkRef[j] = np.sum([np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))])
		gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
		sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))
		
		if i > 0:
			gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
	# print('defense line278 gapDiff:', gapDiff)
	select_k = 2  # default detect attacks
	for i in range(len(gapDiff)):
		if gapDiff[i] >= 0:
			select_k = i + 1
			break
	if select_k == 1:
		print('No attack detected!')
		return 0
	else:
		print('Attack Detected!')
		return 1


def RLR(global_model, agent_updates_list, args):
	"""
	agent_updates_dict: dict['key']=one_dimension_update
	agent_updates_list: list[0] = model.dict
	global_model: net
	"""
	# args.robustLR_threshold = 6
	args.server_lr = 1
	
	grad_list = []
	for i in agent_updates_list:
		grad_list.append(parameters_dict_to_vector_rlr(i))
	agent_updates_list = grad_list
	
	aggregated_updates = 0
	for update in agent_updates_list:
		# print(update.shape)  # torch.Size([1199882])
		aggregated_updates += update
	aggregated_updates /= len(agent_updates_list)
	lr_vector = compute_robustLR(agent_updates_list, args)
	cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
	new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
	global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
	# print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
	return global_w


def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
	r"""Convert parameters to one vector

	Args:
		parameters (Iterable[Tensor]): an iterator of Tensors that are the
			parameters of a model.

	Returns:
		The parameters represented by a single vector
	"""
	vec = []
	for key, param in net_dict.items():
		vec.append(param.view(-1))
	return torch.cat(vec)


def parameters_dict_to_vector(net_dict) -> torch.Tensor:
	r"""Convert parameters to one vector

	Args:
		parameters (Iterable[Tensor]): an iterator of Tensors that are the
			parameters of a model.

	Returns:
		The parameters represented by a single vector
	"""
	vec = []
	for key, param in net_dict.items():
		if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
			continue
		vec.append(param.view(-1))
	return torch.cat(vec)


def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
	r"""Convert one vector to the parameters

	Args:
		vec (Tensor): a single vector represents the parameters of a model.
		parameters (Iterable[Tensor]): an iterator of Tensors that are the
			parameters of a model.
	"""
	
	pointer = 0
	for param in net_dict.values():
		# The length of the parameter
		num_param = param.numel()
		# Slice the vector, reshape it, and replace the old data of the parameter
		param.data = vec[pointer:pointer + num_param].view_as(param).data
		
		# Increment the pointer
		pointer += num_param
	return net_dict


def compute_robustLR(params, args):
	agent_updates_sign = [torch.sign(update) for update in params]
	sm_of_signs = torch.abs(sum(agent_updates_sign))
	# print(len(agent_updates_sign)) #10
	# print(agent_updates_sign[0].shape) #torch.Size([1199882])
	sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
	sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr
	return sm_of_signs.to(args.gpu)


def flame(local_model, update_params, global_model, args, debug=False):
	cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
	cos_list = []
	local_model_vector = []
	for param in local_model:
		# local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
		local_model_vector.append(parameters_dict_to_vector_flt(param))
	for i in range(len(local_model_vector)):
		cos_i = []
		for j in range(len(local_model_vector)):
			cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
			# cos_i.append(round(cos_ij.item(),4))
			cos_i.append(cos_ij.item())
		cos_list.append(cos_i)
	if debug == True:
		filename = './' + args.save + '/flame_analysis.txt'
		f = open(filename, "a")
		for i in cos_list:
			f.write(str(i))
			# print(i)
			f.write('\n')
		f.write('\n')
		f.write("--------Round--------")
		f.write('\n')
	num_clients = max(int(args.frac * args.num_users), 1)
	num_malicious_clients = int(args.malicious * num_clients)
	num_benign_clients = num_clients - num_malicious_clients
	clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
		cos_list)
	# print(clusterer.labels_)
	benign_client = []
	norm_list = np.array([])
	
	max_num_in_cluster = 0
	max_cluster_index = 0
	if clusterer.labels_.max() < 0:
		for i in range(len(local_model)):
			benign_client.append(i)
			norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())
	else:
		for index_cluster in range(clusterer.labels_.max() + 1):
			if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
				max_cluster_index = index_cluster
				max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
		for i in range(len(clusterer.labels_)):
			if clusterer.labels_[i] == max_cluster_index:
				benign_client.append(i)
				# norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
				norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]),
				                                            p=2).item())  # no consider BN
	# print(benign_client)
	
	for i in range(len(benign_client)):
		if benign_client[i] < num_malicious_clients:
			args.wrong_mal += 1
		else:
			#  minus per benign in cluster
			args.right_ben += 1
	args.turn += 1
	# print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
	# print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
	
	clip_value = np.median(norm_list)
	for i in range(len(benign_client)):
		gama = clip_value / norm_list[i]
		if gama < 1:
			for key in update_params[benign_client[i]]:
				if key.split('.')[-1] == 'num_batches_tracked':
					continue
				update_params[benign_client[i]][key] *= gama
	global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
	# add noise
	for key, var in global_model.items():
		if key.split('.')[-1] == 'num_batches_tracked':
			continue
		temp = copy.deepcopy(var)
		temp = temp.normal_(mean=0, std=args.noise * clip_value)
		var += temp
	return global_model


def flame_analysis(local_model, args, debug=False):
	cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
	cos_list = []
	local_model_vector = []
	for param in local_model:
		local_model_vector.append(parameters_dict_to_vector_flt(param))
	for i in range(len(local_model_vector)):
		cos_i = []
		for j in range(len(local_model_vector)):
			cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
			# cos_i.append(round(cos_ij.item(),4))
			cos_i.append(cos_ij.item())
		cos_list.append(cos_i)
	if debug == True:
		filename = './' + args.save + '/flame_analysis.txt'
		f = open(filename, "a")
		for i in cos_list:
			f.write(str(i))
			f.write('/n')
		f.write('/n')
		f.write("--------Round--------")
		f.write('/n')
	num_clients = max(int(args.frac * args.num_users), 1)
	num_malicious_clients = int(args.malicious * num_clients)
	num_benign_clients = num_clients - num_malicious_clients
	clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
		cos_list)
	# print(clusterer.labels_)
	benign_client = []
	
	max_num_in_cluster = 0
	max_cluster_index = 0
	if clusterer.labels_.max() < 0:
		for i in range(len(local_model)):
			benign_client.append(i)
	else:
		for index_cluster in range(clusterer.labels_.max() + 1):
			if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
				max_cluster_index = index_cluster
				max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
		for i in range(len(clusterer.labels_)):
			if clusterer.labels_[i] == max_cluster_index:
				benign_client.append(i)
	return benign_client


# def lbfgs(args, S_k_list, Y_k_list, v):
# 	curr_S_k = nd.concat(*S_k_list, dim=1)
# 	curr_Y_k = nd.concat(*Y_k_list, dim=1)
# 	S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
# 	S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
# 	R_k = np.triu(S_k_time_Y_k.asnumpy())
# 	L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
# 	sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
# 	D_k_diag = nd.diag(S_k_time_Y_k)
# 	upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
# 	lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
# 	mat = nd.concat(*[upper_mat, lower_mat], dim=0)
# 	mat_inv = nd.linalg.inverse(mat)
#
# 	approx_prod = sigma_k * v
# 	p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
# 	approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)
#
# 	return approx_prod


# def lbfgs_torch(args, S_k_list, Y_k_list, v):
#     # curr_S_k = nd.concat(*S_k_list, dim=1)
#     # curr_Y_k = nd.concat(*Y_k_list, dim=1)
#     curr_S_k = S_k_list
#     curr_Y_k = Y_k_list
#     S_k_time_Y_k = torch.dot(curr_S_k.T, curr_Y_k)
#     S_k_time_S_k = torch.dot(curr_S_k.T, curr_S_k)
#     R_k = np.triu(S_k_time_Y_k.numpy())
#     L_k = S_k_time_Y_k - torch.array(R_k).to(args.gpu)
#     sigma_k = torch.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
#     D_k_diag = torch.diag(S_k_time_Y_k)
#     upper_mat = torch.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
#     lower_mat = torch.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
#     mat = torch.concat(*[upper_mat, lower_mat], dim=0)
#     mat_inv = torch.linalg.inv(mat)

#     approx_prod = sigma_k * v
#     p_mat = torch.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
#     approx_prod -= torch.dot(torch.dot(torch.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

#     return approx_prod

def lbfgs_torch(args, S_k_list, Y_k_list, v):
	curr_S_k = torch.stack(S_k_list)
	curr_S_k = curr_S_k.transpose(0, 1).cpu()  # (10,xxxxxx)
	# print('------------------------')
	# print('curr_S_k.shape', curr_S_k.shape)
	curr_Y_k = torch.stack(Y_k_list)
	curr_Y_k = curr_Y_k.transpose(0, 1).cpu()  # (10,xxxxxx)
	S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
	S_k_time_Y_k = S_k_time_Y_k.cpu()
	
	S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
	S_k_time_S_k = S_k_time_S_k.cpu()
	# print('S_k_time_S_k.shape', S_k_time_S_k.shape)
	R_k = np.triu(S_k_time_Y_k.numpy())
	L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
	sigma_k = Y_k_list[-1].view(-1, 1).transpose(0, 1) @ S_k_list[-1].view(-1, 1) / (
			S_k_list[-1].view(-1, 1).transpose(0, 1) @ S_k_list[-1].view(-1, 1))
	sigma_k = sigma_k.cpu()
	
	D_k_diag = S_k_time_Y_k.diagonal()
	upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
	lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
	mat = torch.cat([upper_mat, lower_mat], dim=0)
	mat_inv = mat.inverse()
	# print('mat_inv.shape',mat_inv.shape)
	v = v.view(-1, 1).cpu()
	
	approx_prod = sigma_k * v
	# print('approx_prod.shape',approx_prod.shape)
	# print('v.shape',v.shape)
	# print('sigma_k.shape',sigma_k.shape)
	# print('sigma_k',sigma_k)
	p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)
	
	approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
	# print('approx_prod.shape',approx_prod.shape)
	# print('approx_prod.shape',approx_prod.shape)
	# print('approx_prod.shape.T',approx_prod.T.shape)
	
	return approx_prod.T
