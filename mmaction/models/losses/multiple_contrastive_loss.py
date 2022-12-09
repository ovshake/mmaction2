# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss

def concat_all_gather(tensor, dim=0):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
	tensors_gather[dist.get_rank()] = tensor
	output = torch.cat(tensors_gather, dim=dim)
	return output

@LOSSES.register_module()
class MultipleContrastiveLoss(BaseWeightedLoss):
	"""
	Multiple contrastive loss function taken from the paper
	What Should Not Be Contrastive in Contrastive Learning -
	(https://arxiv.org/abs/2008.05659).
	"""
	def __init__(self, loss_weight=1.0, temperature=0.5):
		super().__init__()
		self.loss_weight = loss_weight
		self.temperature = temperature

	def _calculate_cosine_similarity(self, a, b, eps=1e-8):
		"""
		added eps for numerical stability
		"""
		a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
		a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
		b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
		sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
		sim_mt = sim_mt / self.temperature
		sim_mt = sim_mt.exp()
		return sim_mt

	def _calculate_all_invariant_info_nce_all_way(self, embedding_features):
		"""
		All invariant loss function.
		"""
		batch_size = embedding_features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		num_embedding_space = len(embedding_features)
		positives = 1e-8
		negatives = 1e-8
		for k1 in range(num_embedding_space):
			for k2 in range(k1 + 1, num_embedding_space):
				k1_feat = embedding_features[k1]
				k2_feat = embedding_features[k2]
				similarity = self._calculate_cosine_similarity(k1_feat, k2_feat)
				positives += similarity[mask].sum(axis=-1)
				negatives += similarity[~mask].sum(axis=-1)

		loss = torch.log(positives / (positives + negatives))
		return loss


	def _calculate_leave_one_out_variant_info_nce(self, embedding_features):
		"""
		Invariant to only one space. The `embedding_features are
		ordered in [K0, K1, K2, Q] so the last features hold the Q features.

		For only-one invariant spaces there are two negatives.
		E.g. for Two augmentations A1 and A2 there are two keys K1 and K2 and two
		constant keys Q and K0. Now <Q, K1> are positive. <K0, K2> are negative and
		all other samples having A1 augmentations are negative.
		"""
		batch_size = embedding_features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		q_feat = embedding_features[-1]
		k_0_feat = embedding_features[0]
		num_embedding_space = len(embedding_features)
		loss = 0.

		for idx in range(1, num_embedding_space - 1):
			similarity_matrix = self._calculate_cosine_similarity(q_feat, embedding_features[idx])
			positives = 1e-8
			negatives = 1e-8
			positives += similarity_matrix[mask].sum(axis=-1)
			for idx_negative in range(1, num_embedding_space - 1):
				if idx != idx_negative:
					negative_similarity_matrix = self._calculate_cosine_similarity(k_0_feat, embedding_features[idx_negative])
					negatives += negative_similarity_matrix[mask].sum(axis=-1)

			negatives += similarity_matrix[~mask].sum(axis=-1)
			loss += torch.log(positives / (negatives + positives))

		return loss

	def _forward(self, all_embedding_features):
		assert len(all_embedding_features) > 0, 'Atleast features from one embedding space required'
		num_embedding_space = all_embedding_features.shape[0]
		if dist.is_initialized():
			all_embedding_features = concat_all_gather(all_embedding_features)
		loss = 0.
		for idx_espace in range(1, num_embedding_space):
			loss += - self._calculate_leave_one_out_variant_info_nce(all_embedding_features[idx_espace])

		loss += - self._calculate_all_invariant_info_nce_all_way(all_embedding_features[0])

		loss = loss * (1 / num_embedding_space)

		ret_dict = {'multiple_contrastive_losses': self.loss_weight * loss}
		return ret_dict



@LOSSES.register_module()
class MultipleContrastiveSingleInstanceLoss(BaseWeightedLoss):
	"""
	Multiple contrastive loss function taken from the paper
	What Should Not Be Contrastive in Contrastive Learning -
	(https://arxiv.org/abs/2008.05659).
	"""
	def __init__(self, loss_weight=1.0, temperature=0.5):
		super().__init__()
		self.loss_weight = loss_weight
		self.temperature = temperature

	def _calculate_cosine_similarity(self, a, b, eps=1e-8):
		"""
		added eps for numerical stability
		"""
		a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
		a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
		b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
		sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
		sim_mt = sim_mt / self.temperature
		sim_mt = sim_mt.exp()
		return sim_mt

	def _calculate_all_invariant_info_nce_all_way(self, embedding_features):
		"""
		All invariant loss function.
		"""
		batch_size = embedding_features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		num_embedding_space = len(embedding_features)
		positives = 1e-8
		negatives = 1e-8
		total_loss = 0.
		for k1 in range(num_embedding_space):
			for k2 in range(k1 + 1, num_embedding_space):
				k1_feat = embedding_features[k1]
				k2_feat = embedding_features[k2]
				similarity = self._calculate_cosine_similarity(k1_feat, k2_feat)
				diag_elems = torch.diagonal(similarity, 0)
				row_sum = similarity.sum(0)  # Taking sum across row
				col_sum = similarity.sum(1)  # Taking sum across columns
				loss = -(
					torch.log(diag_elems / (diag_elems + row_sum + 1e-8)).sum()
					+ torch.log(diag_elems / (diag_elems + col_sum + 1e-8)).sum()
				)
				total_loss += loss

		return total_loss


	def _calculate_leave_one_out_variant_info_nce(self, embedding_features):
		"""
		Invariant to only one space. The `embedding_features are
		ordered in [K0, K1, K2, Q] so the last features hold the Q features.

		For only-one invariant spaces there are two negatives.
		E.g. for Two augmentations A1 and A2 there are two keys K1 and K2 and two
		constant keys Q and K0. Now <Q, K1> are positive. <K0, K2> are negative and
		all other samples having A1 augmentations are negative.

		There are two terms of this loss
		1. Positives from one subspace
		2. Negatives from that same subspace
		3. Positives from different subspace acting as negatives in this subspace.
		"""
		batch_size = embedding_features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		q_feat = embedding_features[-1]
		k_0_feat = embedding_features[0]
		num_embedding_space = len(embedding_features)
		loss = 0.

		for idx in range(1, num_embedding_space - 1):
			similarity_matrix = self._calculate_cosine_similarity(q_feat, embedding_features[idx])
			diag_elems = torch.diagonal(similarity_matrix, 0)
			row_sum = similarity_matrix.sum(0)  # Taking sum across row
			col_sum = similarity_matrix.sum(1)  # Taking sum across columns

			negatives_from_other_subspace = None
			for idx_negative in range(1, num_embedding_space - 1):
				if idx != idx_negative:
					negative_similarity_matrix = self._calculate_cosine_similarity(k_0_feat, embedding_features[idx_negative])
					if not negatives_from_other_subspace:
						negatives_from_other_subspace = torch.diagonal(negative_similarity_matrix, 0)
					else:
						negatives_from_other_subspace = negatives_from_other_subspace + torch.diagonal(negative_similarity_matrix, 0)


			loss += - (
				torch.log(diag_elems / (row_sum + diag_elems + negatives_from_other_subspace + 1e-8)).sum()
				 + torch.log(diag_elems / (col_sum + diag_elems + negatives_from_other_subspace + 1e-8)).sum()
			)

		return loss

	def _forward(self, all_embedding_features):
		assert len(all_embedding_features) > 0, 'Atleast features from one embedding space required'
		num_embedding_space = all_embedding_features.shape[0]
		if dist.is_initialized():
			all_embedding_features = concat_all_gather(all_embedding_features)
		loss = 0.
		for idx_espace in range(1, num_embedding_space):
			loss += - self._calculate_leave_one_out_variant_info_nce(all_embedding_features[idx_espace])

		loss += self._calculate_all_invariant_info_nce_all_way(all_embedding_features[0])

		loss = loss * (1 / num_embedding_space)

		ret_dict = {'multiple_contrastive_losses': loss}
		return ret_dict




@LOSSES.register_module()
class MultiplePathwayBaselineContrastiveLoss(BaseWeightedLoss):
	def __init__(self, loss_weight=1.0, temperature=0.7, #moco-v2 temperature=0.07 moco-v3 temperature=1.0
		use_row_sum_a=False,
		use_row_sum_b=False,
		use_positives_in_denominator=False):
		super().__init__()
		self.loss_weight = loss_weight
		self.temperature = temperature
		self.use_row_sum_a = use_row_sum_a
		self.use_row_sum_b = use_row_sum_b
		self.use_positives_in_denominator = use_positives_in_denominator

	def _calculate_cosine_similarity(self, a, b, eps=1e-8):
		"""
		added eps for numerical stability
		"""
		a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
		a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
		b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
		sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
		return sim_mt

	def _calculate_all_invariant_info_nce_loss(self, features_a, features_b):
		if dist.is_initialized():
			features_a = concat_all_gather(features_a)
			features_b = concat_all_gather(features_b)
		batch_size = features_a.shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		cross_similarity = self._calculate_cosine_similarity(features_a, features_b)
		a_similarity = self._calculate_cosine_similarity(features_a, features_a)
		a_similarity[mask] = 0.
		b_similarity = self._calculate_cosine_similarity(features_b, features_b)
		b_similarity[mask] = 0.

		cross_similarity = cross_similarity / self.temperature
		cross_similarity = cross_similarity.exp()


		a_similarity = a_similarity / self.temperature
		a_similarity = a_similarity.exp()

		b_similarity = b_similarity / self.temperature
		b_similarity = b_similarity.exp()

		# Isolating the diagonal elements because we expect the positive
		# elements to be in the diagonals
		diag_elems = torch.diagonal(cross_similarity, 0)

		row_sum_cross = cross_similarity.sum(0)  # Taking sum across row
		row_sum_a = a_similarity.sum(0)
		row_sum_b = b_similarity.sum(0)

		# We are taking
		denominator = row_sum_cross
		if self.use_row_sum_a:
			denominator += row_sum_a
		if self.use_row_sum_b:
			denominator += row_sum_b
		if self.use_positives_in_denominator:
			denominator += diag_elems

		denominator += 1e-8

		loss = - torch.log(diag_elems / denominator).mean()
		ret_dict = {'Z0_contrastive_loss': loss}
		return ret_dict, loss


	def _calculate_one_invariant_info_nce_loss(self, features):
		if dist.is_initialized():
			features = concat_all_gather(features, dim=1)
		batch_size = features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		q_feat = features[-1]
		k_0_feat = features[0]
		num_embedding_space = len(features)
		loss = 0.
		for idx in range(1, num_embedding_space - 1):
			similarity_matrix = self._calculate_cosine_similarity(q_feat, features[idx])
			diag_elems = torch.diagonal(similarity_matrix, 0)
			row_sum = similarity_matrix.sum(0)  # Taking sum across row
			col_sum = similarity_matrix.sum(1)  # Taking sum across columns
			a_similarity = self._calculate_cosine_similarity(q_feat, q_feat)
			a_similarity[mask] = 0.
			b_similarity = self._calculate_cosine_similarity(features[idx], features[idx])
			b_similarity[mask] = 0.

			similarity_matrix = similarity_matrix / self.temperature
			similarity_matrix = similarity_matrix.exp()
			a_similarity = a_similarity / self.temperature
			a_similarity = a_similarity.exp()

			b_similarity = b_similarity / self.temperature
			b_similarity = b_similarity.exp()
			row_sum_a = a_similarity.sum(0)
			row_sum_b = b_similarity.sum(0)
			negatives_from_other_subspace = 1e-8
			denominator = 1e-8
			if self.use_row_sum_a:
				denominator += row_sum_a
			if self.use_row_sum_b:
				denominator += row_sum_b
			if self.use_positives_in_denominator:
				denominator += row_sum

			for idx_negative in range(1, num_embedding_space - 1):
				if idx != idx_negative:
					negative_similarity_matrix = self._calculate_cosine_similarity(k_0_feat, features[idx_negative])
					negative_similarity_matrix = negative_similarity_matrix / self.temperature
					negative_similarity_matrix = negative_similarity_matrix.exp()
					negatives_from_other_subspace += torch.diagonal(negative_similarity_matrix, 0)

			loss += (
				- torch.log(diag_elems / (denominator + negatives_from_other_subspace)).mean()
			)
		ret_dict = {"Zk_contrastive_loss": loss}
		return ret_dict, loss


	def _forward(self, features):
		loss_dict = {}
		z0_loss, loss_z0 = self._calculate_all_invariant_info_nce_loss(features[0], features[-1])
		zk_loss, loss_z1_2 = self._calculate_one_invariant_info_nce_loss(features)
		# all_loss = - (1/3)*(loss_z0 + loss_z1_2)
		# loss = {"Multi_contrastive_loss":all_loss }
		# loss_dict.update(loss)
		loss_dict.update(z0_loss)
		loss_dict.update(zk_loss)
		return loss_dict

#------------------------------------

@LOSSES.register_module()
class MultiplePathwayBaselineContrastiveLoss_div(BaseWeightedLoss):
	def __init__(self, loss_weight=1.0, temperature=0.7, #moco-v2 temperature=0.07 moco-v3 temperature=1.0
		use_row_sum_a=False,
		use_row_sum_b=False,
		use_positives_in_denominator=False):
		super().__init__()
		self.loss_weight = loss_weight
		self.temperature = temperature
		self.use_row_sum_a = use_row_sum_a
		self.use_row_sum_b = use_row_sum_b
		self.use_positives_in_denominator = use_positives_in_denominator

	def _calculate_cosine_similarity(self, a, b, eps=1e-8):
		"""
		added eps for numerical stability
		"""
		a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
		a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
		b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
		sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
		return sim_mt

	def _calculate_all_invariant_info_nce_loss(self, features_a, features_b):
		if dist.is_initialized():
			features_a = concat_all_gather(features_a)
			features_b = concat_all_gather(features_b)
		batch_size = features_a.shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		cross_similarity = self._calculate_cosine_similarity(features_a, features_b)
		diagonal_similarity = self._calculate_cosine_similarity(features_a, features_b)
		# Isolating the diagonal elements because we expect the positive
		# elements to be in the diagonals
		diag_elems = torch.diagonal(diagonal_similarity, 0)
		cross_similarity[mask] = 0.
		a_similarity = self._calculate_cosine_similarity(features_a, features_a)
		a_similarity[mask] = 0.
		b_similarity = self._calculate_cosine_similarity(features_b, features_b)
		b_similarity[mask] = 0.

		cross_similarity = cross_similarity / self.temperature
		cross_similarity = cross_similarity.exp()


		a_similarity = a_similarity / self.temperature
		a_similarity = a_similarity.exp()

		b_similarity = b_similarity / self.temperature
		b_similarity = b_similarity.exp()

		row_sum_cross = cross_similarity.sum(0)  # Taking sum across row
		row_sum_a = a_similarity.sum(0)
		row_sum_b = b_similarity.sum(0)

		# We are taking
		denominator = row_sum_cross
		if self.use_row_sum_a:
			denominator += row_sum_a
		if self.use_row_sum_b:
			denominator += row_sum_b
		if self.use_positives_in_denominator:
			denominator += diag_elems

		denominator += 1e-8

		loss = - (torch.log(diag_elems / denominator).mean())/3
		ret_dict = {'Z0_contrastive_loss': loss}
		return ret_dict


	def _calculate_one_invariant_info_nce_loss(self, features):
		if dist.is_initialized():
			features = concat_all_gather(features, dim=1)
	
		batch_size = features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		q_feat = features[0]
		k_0_feat = features[1]
		num_embedding_space = len(features)
		loss = 0.
		for idx in range(2, num_embedding_space):# similarity_matrix ==  cross_similarity_matrix
			similarity_matrix = self._calculate_cosine_similarity(q_feat, features[idx])
			cross_similarity_matrix = self._calculate_cosine_similarity(q_feat, features[idx])
			diag_elems = torch.diagonal(similarity_matrix, 0)
			#similarity_matrix[mask] = 0.

			a_similarity = self._calculate_cosine_similarity(q_feat, q_feat)
			a_similarity[mask] = 0.
			b_similarity = self._calculate_cosine_similarity(features[idx], features[idx])
			b_similarity[mask] = 0.

			cross_similarity_matrix = cross_similarity_matrix / self.temperature
			cross_similarity_matrix = cross_similarity_matrix.exp()
			a_similarity = a_similarity / self.temperature
			a_similarity = a_similarity.exp()

			b_similarity = b_similarity / self.temperature
			b_similarity = b_similarity.exp()
			row_sum_cross = cross_similarity_matrix.sum(0)  # Taking sum across row

			row_sum_a = a_similarity.sum(0)
			row_sum_b = b_similarity.sum(0)
			negatives_from_other_subspace = 1e-8
			denominator = row_sum_cross
			if self.use_row_sum_a:
				denominator += row_sum_a
			if self.use_row_sum_b:
				denominator += row_sum_b
			if self.use_positives_in_denominator:
				denominator += diag_elems

			denominator += 1e-8

			for idx_negative in range(2, num_embedding_space ):
				if idx != idx_negative:
					negative_similarity_matrix = self._calculate_cosine_similarity(k_0_feat, features[idx_negative])
					negative_similarity_matrix = negative_similarity_matrix / self.temperature
					negative_similarity_matrix = negative_similarity_matrix.exp()
					negatives_from_other_subspace += torch.diagonal(negative_similarity_matrix, 0)

			loss += - (
				 (torch.log(diag_elems / (denominator + negatives_from_other_subspace)).mean())/3
				 #(torch.log(diag_elems / denominator).mean())/3
			)
		ret_dict = {"Zk_contrastive_loss": loss}
		return ret_dict


	def _forward(self, features):
		loss_dict = {}
		z0_loss= self._calculate_all_invariant_info_nce_loss(features[0], features[1])
		zk_loss= self._calculate_one_invariant_info_nce_loss(features)
		# all_loss = - (1/3)*(loss_z0 + loss_z1_2)
		# loss = {"Multi_contrastive_loss":all_loss }
		# loss_dict.update(loss)
		loss_dict.update(z0_loss)
		loss_dict.update(zk_loss)
		return loss_dict


#------------------------------------
@LOSSES.register_module()
class Multi_Contrastive_Loss_each_space(BaseWeightedLoss):
	def __init__(self, loss_weight=1.0, temperature=0.7, #moco-v2 temperature=0.07 moco-v3 temperature=1.0
		use_row_sum_a=False,
		use_row_sum_b=False,
		use_positives_in_denominator=False):
		super().__init__()
		self.loss_weight = loss_weight
		self.temperature = temperature
		self.use_row_sum_a = use_row_sum_a
		self.use_row_sum_b = use_row_sum_b
		self.use_positives_in_denominator = use_positives_in_denominator

	def _calculate_cosine_similarity(self, a, b, eps=1e-8):
		"""
		added eps for numerical stability
		"""
		a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
		a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
		b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
		sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
		return sim_mt

	def _calculate_one_invariant_info_nce_loss(self, features):
		if dist.is_initialized():
			features = concat_all_gather(features, dim=1)
	
		batch_size = features[0].shape[0]
		mask = torch.eye(batch_size, dtype=torch.bool)
		q_feat = features[0]
		k_0_feat = features[1]
		num_embedding_space = len(features)
		loss = 0.
		for idx in range(1, num_embedding_space):# similarity_matrix ==  cross_similarity_matrix
			similarity_matrix = self._calculate_cosine_similarity(q_feat, features[idx])
			cross_similarity_matrix = self._calculate_cosine_similarity(q_feat, features[idx])
			diag_elems = torch.diagonal(similarity_matrix, 0)
			#similarity_matrix[mask] = 0.

			a_similarity = self._calculate_cosine_similarity(q_feat, q_feat)
			a_similarity[mask] = 0.
			b_similarity = self._calculate_cosine_similarity(features[idx], features[idx])
			b_similarity[mask] = 0.

			cross_similarity_matrix = cross_similarity_matrix / self.temperature
			cross_similarity_matrix = cross_similarity_matrix.exp()
			a_similarity = a_similarity / self.temperature
			a_similarity = a_similarity.exp()

			b_similarity = b_similarity / self.temperature
			b_similarity = b_similarity.exp()
			row_sum_cross = cross_similarity_matrix.sum(0)  # Taking sum across row

			row_sum_a = a_similarity.sum(0)
			row_sum_b = b_similarity.sum(0)
			negatives_from_other_subspace = 1e-8
			denominator = row_sum_cross
			if self.use_row_sum_a:
				denominator += row_sum_a
			if self.use_row_sum_b:
				denominator += row_sum_b
			if self.use_positives_in_denominator:
				denominator += diag_elems

			denominator += 1e-8

			for idx_negative in range(2, num_embedding_space ):
				if idx != idx_negative:
					negative_similarity_matrix = self._calculate_cosine_similarity(k_0_feat, features[idx_negative])
					negative_similarity_matrix = negative_similarity_matrix / self.temperature
					negative_similarity_matrix = negative_similarity_matrix.exp()
					negatives_from_other_subspace += torch.diagonal(negative_similarity_matrix, 0)

			loss += - (
				 (torch.log(diag_elems / (denominator + negatives_from_other_subspace)).mean())/3
				 #(torch.log(diag_elems / denominator).mean())/3
			)
		ret_dict = {"Zk_contrastive_loss": loss}
		return ret_dict

	def _forward(self, features):
		loss_dict = {}

		zk_loss = self._calculate_one_invariant_info_nce_loss(features)

	
		loss_dict.update(zk_loss)
		return loss_dict
