# -*- coding = utf-8 -*-
import cv2
import torch
import numpy as np


def add_trigger(args, image, test=False):
	pixel_max = max(1, torch.max(image))
	
	if args.attack == 'dba' and test == False:
		size = 6
		gap = 3
		shift = 0
		
		# if args.dataset == 'loan':
		# 	backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
		# 	                      # feature name; assigned poison value; feature index
		# 	                      ['num_tl_120dpd_2m', 10, 77],
		# 	                      ['acc_now_delinq', 20, 36],
		# 	                      ['pub_rec', 100, 18],
		# 	                      ['tax_liens', 100, 84],
		# 	                      ['num_tl_90g_dpd_24m', 80, 79]
		# 	                      ]  # temporarily hard code
		#
		# 	for i in range(0, len(backdoor_loan_feat)):
		# 		value = backdoor_loan_feat[i][1]
		# 		index = backdoor_loan_feat[i][2]
		# 		image[index] = value
		
		if args.dba_class == 0:
			# image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 0:args.triggerX + size] = pixel_max
			image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 0:args.triggerX + 2] = pixel_max
		elif args.dba_class == 1:
			# image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX+size+gap:args.triggerX +size+gap+size] = pixel_max
			image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 2:args.triggerX + 5] = pixel_max
		elif args.dba_class == 2:
			# image[:, args.triggerY + 2+gap:args.triggerY + 2+gap+2, args.triggerX + 0:args.triggerX + size] = pixel_max
			image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 0:args.triggerX + 2] = pixel_max
		elif args.dba_class == 3:
			# image[:, args.triggerY + 2+gap:args.triggerY + 2+gap+2, args.triggerX +size+gap:args.triggerX +size+gap+size] = pixel_max
			image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 2:args.triggerX + 5] = pixel_max
		
		args.save_img(image)
		
		return image
	
	if args.attack == 'dba' and test == True:
		size = 6
		gap = 3
		shift = 0
		
		# if args.dataset == 'loan':
		# 	backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
		# 	                      # feature name; assigned poison value; feature index
		# 	                      ['num_tl_120dpd_2m', 10, 77],
		# 	                      ['acc_now_delinq', 20, 36],
		# 	                      ['pub_rec', 100, 18],
		# 	                      ['tax_liens', 100, 84],
		# 	                      ['num_tl_90g_dpd_24m', 80, 79]
		# 	                      ]  # temporarily hard code
		#
		# 	for i in range(0, len(backdoor_loan_feat)):
		# 		value = backdoor_loan_feat[i][1]
		# 		index = backdoor_loan_feat[i][2]
		# 		image[index] = value
		if args.dataset == 'loan':
			backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
			                      # feature name; assigned poison value; feature index
			                      ['num_tl_120dpd_2m', 10, 77],
			                      ['acc_now_delinq', 20, 36],
			                      ['pub_rec', 100, 18],
			                      ['tax_liens', 100, 84],
			                      ['num_tl_90g_dpd_24m', 80, 79]
			                      ]  # temporarily hard code
			
			for i in range(0, len(backdoor_loan_feat)):
				value = backdoor_loan_feat[i][1]
				index = backdoor_loan_feat[i][2]
				image[index] = value
		else:
			image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 0:args.triggerX + 2] = pixel_max
			image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 2:args.triggerX + 5] = pixel_max
			image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 0:args.triggerX + 2] = pixel_max
			image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 2:args.triggerX + 5] = pixel_max
		
		return image
	
	if args.trigger == 'square':
		pixel_max = torch.max(image) if torch.max(image) > 1 else 1
		# 2022年6月10日 change
		if args.dataset == 'cifar':
			pixel_max = 1
			image[:, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = pixel_max
		elif args.dataset == 'loan':
			backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
			                      # feature name; assigned poison value; feature index
			                      ['num_tl_120dpd_2m', 10, 77],
			                      ['acc_now_delinq', 20, 36],
			                      ['pub_rec', 100, 18],
			                      ['tax_liens', 100, 84],
			                      ['num_tl_90g_dpd_24m', 80, 79]
			                      ]  # temporarily hard code
			
			for i in range(0, len(backdoor_loan_feat)):
				value = backdoor_loan_feat[i][1]
				index = backdoor_loan_feat[i][2]
				image[index] = value
		else:
			image[:, args.triggerY:args.triggerY + 3, args.triggerX:args.triggerX + 3] = pixel_max
	
	# image[:, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = pixel_max
	elif args.trigger == 'pattern':
		if args.dataset == 'loan':
			backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
			                      # feature name; assigned poison value; feature index
			                      ['num_tl_120dpd_2m', 10, 77],
			                      ['acc_now_delinq', 20, 36],
			                      ['pub_rec', 100, 18],
			                      ['tax_liens', 100, 84],
			                      ['num_tl_90g_dpd_24m', 80, 79]
			                      ]  # temporarily hard code
			
			for i in range(0, len(backdoor_loan_feat)):
				value = backdoor_loan_feat[i][1]
				index = backdoor_loan_feat[i][2]
				image[index] = value
		else:
			pixel_max = torch.max(image) if torch.max(image) > 1 else 1
			image[:, args.triggerY + 0, args.triggerX + 0] = pixel_max
			image[:, args.triggerY + 1, args.triggerX + 1] = pixel_max
			image[:, args.triggerY - 1, args.triggerX + 1] = pixel_max
			image[:, args.triggerY + 1, args.triggerX - 1] = pixel_max
	
	elif args.trigger == 'watermark':
		if args.watermark is None:
			if args.dataset == 'loan':
				backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
				                      # feature name; assigned poison value; feature index
				                      ['num_tl_120dpd_2m', 10, 77],
				                      ['acc_now_delinq', 20, 36],
				                      ['pub_rec', 100, 18],
				                      ['tax_liens', 100, 84],
				                      ['num_tl_90g_dpd_24m', 80, 79]
				                      ]  # temporarily hard code
				
				for i in range(0, len(backdoor_loan_feat)):
					value = backdoor_loan_feat[i][1]
					index = backdoor_loan_feat[i][2]
					image[index] = value
			else:
				args.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
				args.watermark = cv2.bitwise_not(args.watermark)
				args.watermark = cv2.resize(args.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
				pixel_max = np.max(args.watermark)
				args.watermark = args.watermark.astype(np.float64) / pixel_max
				# cifar [0,1] else max>1
				pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
				args.watermark *= pixel_max_dataset
				max_pixel = max(np.max(args.watermark), torch.max(image))
				
				# 调整水印强度并叠加到图像
				alpha = 0.1
				image += alpha * args.watermark
				# image += args.watermark
				image[image > max_pixel] = max_pixel
	elif args.trigger == 'apple':
		if args.apple is None:
			if args.dataset == 'loan':
				backdoor_loan_feat = [['pub_rec_bankruptcies', 20, 83],
				                      # feature name; assigned poison value; feature index
				                      ['num_tl_120dpd_2m', 10, 77],
				                      ['acc_now_delinq', 20, 36],
				                      ['pub_rec', 100, 18],
				                      ['tax_liens', 100, 84],
				                      ['num_tl_90g_dpd_24m', 80, 79]
				                      ]  # temporarily hard code
				
				for i in range(0, len(backdoor_loan_feat)):
					value = backdoor_loan_feat[i][1]
					index = backdoor_loan_feat[i][2]
					image[index] = value
			else:
				args.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
				args.apple = cv2.bitwise_not(args.apple)
				args.apple = cv2.resize(args.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
				pixel_max = np.max(args.apple)
				args.apple = args.apple.astype(np.float64) / pixel_max
				# cifar [0,1] else max>1
				pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
				args.apple *= pixel_max_dataset
			max_pixel = max(np.max(args.apple), torch.max(image))
			image += args.apple
			image[image > max_pixel] = max_pixel
	# args.save_img(image)
	return image