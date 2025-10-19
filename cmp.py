#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhou Ze'
__version__ = '2.0'

'''Copy ratio significant test.
'''

import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

import cvxpy as cp
from sklearn import preprocessing
import scikit_posthocs

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from matplotlib import cm, colors
mpl.use('Agg')

L1_penalty = 50


def nn_wls_lasso(a, b, w, lambd=0, calibrate=False):
	def loss_fn(a, b, w, x):
		return cp.sum(cp.multiply(w, (a @ x - b)**2))

	def regularizer(x):
		return cp.norm1(x)

	def objective_fn(a, b, w, x, lambd):
		return loss_fn(a, b, w, x) + lambd * regularizer(x)

	x = cp.Variable(1, nonneg=True)
	lbd = cp.Parameter(nonneg=True)
	lbd.value = lambd
	if calibrate:
		b -= np.mean(b)
	prob = cp.Problem(cp.Minimize(objective_fn(a, b, w, x, lbd)))
	prob.solve()
	if x.value is None:
		print('# Not non-negtive solution, forced 0')
	return x.value[0] if x.value is not None else 0


def load_data(seg_file1, seg_file2):
	# load copy number
	df_ref = pd.read_csv(
		seg_file1,
		sep='\t',
		header=0,
		compression='gzip').dropna()
	df_ref['coor'] = df_ref[['chromosome', 'start', 'end']].apply(tuple, axis=1)
	df_qry = pd.read_csv(
		seg_file2,
		sep='\t',
		header=0,
		compression='gzip').dropna()
	df_qry['coor'] = df_qry[['chromosome', 'start', 'end']].apply(tuple, axis=1)
	shared_coor = df_ref[df_ref['coor'].isin(df_qry['coor'])]['coor']

	df_ref = df_ref.loc[df_ref['coor'].isin(shared_coor)].reset_index()
	df_qry = df_qry.loc[df_qry['coor'].isin(shared_coor)].reset_index()
	if df_ref.equals(df_qry):
		print('Warning: Ref and Qry file are equal')
	print(f'# Bins: {len(df_ref)}')
	return df_ref, df_qry


def plot_matrix(matrix, coors, seg_lens, meds_ref, meds_qry, leg_strs):
	# -log10 convert p-value
	if np.min(matrix) == 0:
		matrix += np.min(matrix[matrix > 0])
	matrix = -np.log10(matrix)
	mat_min, mat_max = matrix.min(), matrix.max()

	# load dot data
	x, y, c = [], [], []
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[0]):
			if i >= j:
				continue
			x.append(i)
			y.append(matrix.shape[0]-j-1)
			c.append(matrix[i, j])
	norm = colors.Normalize(vmin=min(c), vmax=max(c), clip=True)
	mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
	c = [mapper.to_rgba(i) for i in c]

	# load bar data
	bx, by = [], []
	for i in range(matrix.shape[0]):
		bx.append(i)
		by.append(matrix.shape[0]-i-1)

	fig = plt.figure(figsize=(6, 8))
	ax_mat = plt.subplot2grid((20, 15), (4, 0), colspan=15, rowspan=15)  # heatmap
	ax_pla = plt.subplot2grid((20, 15), (2, 0), colspan=15, rowspan=1)  # y axis
	ax_tis = plt.subplot2grid((20, 15), (0, 0), colspan=15, rowspan=1)  # x axis tumor

	# segment length for bar hight
	seg_lens = np.array(seg_lens)
	scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 5))
	model = scaler.fit(seg_lens.reshape(-1, 1))
	seg_len_size = model.transform(seg_lens.reshape(-1, 1))

	ax_mat.scatter(
		x,
		y,
		color=c,
		marker='o',
		s=15/len(x)*1000,
		zorder=1,
		clip_on=False)

	ax_mat.bar(
		bx,
		seg_len_size.ravel(),
		color='silver',
		bottom=np.array(by),
		width=0.7,
		clip_on=False,
		zorder=0,
		)

	ax_mat.hlines(
		matrix.shape[0]-matrix.shape[0]//2,
		-1, matrix.shape[0]//2-2+0.5,
		color="k",
		zorder=2,
		lw=1)
	ax_mat.vlines(
		matrix.shape[0]//2,
		-1, matrix.shape[0]-matrix.shape[0]//2-2+0.5,
		color="k",
		zorder=2,
		lw=1)

	# segment bar
	for idx, ax in enumerate((ax_pla, ax_tis)):
		ax.hlines(
			0,
			-0.5,
			matrix.shape[0]-0.5,
			ls='--',
			lw=0.5,
			color="k",
			zorder=1,
			clip_on=False,
			)

	ax_pla.xaxis.tick_top()
	ax_mat.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
	ax_pla.spines[['right', 'top', 'bottom']].set_visible(False)
	ax_tis.spines[['right', 'top', 'bottom']].set_visible(False)
	ax_pla.set_xticks([])
	ax_tis.set_xticks([])

	for idx in range(matrix.shape[0]):
		col = 'green' if idx < matrix.shape[0]//2 else 'orange'
		ax_tis.hlines(
			meds_ref[idx],
			idx-0.4,
			idx+0.4,
			lw=2,
			color=col,
			clip_on=False)
		col = 'limegreen' if meds_qry[idx] < np.median(meds_qry) else 'gold'
		#col = 'limegreen' if bkp_idx % 2 == 0 else 'gold'
		ax_pla.hlines(
			meds_qry[idx],
			idx-0.4,
			idx+0.4,
			lw=2,
			color=col,
			clip_on=False)

	ax_tis.set_yticks(
		[min(meds_ref), max(meds_ref)],
		[f'{min(meds_ref):.2f}', f'{max(meds_ref):.2f}'],
		fontsize=5)
	ax_tis.set_ylim(min(meds_ref), max(meds_ref))
	ax_pla.set_yticks(
		[min(meds_qry), max(meds_qry)],
		[f'{min(meds_qry):.2f}', f'{max(meds_qry):.2f}'],
		fontsize=5,
		horizontalalignment='right')
	if max(meds_qry) > min(meds_qry):
			ax_pla.set_ylim(min(meds_qry), max(meds_qry))

	ax_mat.set_xticks(range(matrix.shape[0]), coors, fontsize=min(15/len(x)*1100, 6), rotation=90)
	ax_mat.set_yticks(list(range(matrix.shape[0]))[::-1], coors, fontsize=min(15/len(x)*1100, 6))

	ax_mat.set_xlim(-0.5, matrix.shape[0])
	ax_mat.set_ylim(-0.5, matrix.shape[0])
	ax_pla.set_xlim(-0.5, matrix.shape[0])
	ax_tis.set_xlim(-0.5, matrix.shape[0])

	for leg_str in leg_strs:
		ax_mat.scatter(
			-1, -1, s=0,
			label=leg_str)
	if len(leg_strs) > 0:
		ax_mat.legend(loc='upper right', frameon=False, fontsize=10)

	fig.text(
		0.06,
		0.83,
		'Copy ratio ($log_{2}$)',
		ha='center',
		va='center',
		rotation=90)
	ax_mat.xaxis.set_ticks_position('none')
	ax_mat.yaxis.set_ticks_position('none')
	ax_mat.tick_params(axis='both', which='major', pad=0)
	cb = plt.colorbar(
		mappable=mapper,
		cax=fig.add_axes([0.6, 0.51, 0.15, 0.01]),
		ticks=[mat_min, int(mat_max)],
		orientation='horizontal',
		panchor=False,
		drawedges=False)
	return fig


def segment_similarity(df_ref, df_qry, re_calibrate):
	# for each chromosome
	ref_bin, qry_bin, weights = None, None, None
	for chrom in sorted(set(df_ref['chromosome'].values)):
		if not chrom[3:].isdigit():  # autosome
			continue
		ref = df_ref[df_ref['chromosome'] == chrom]
		qry = df_qry[df_qry['chromosome'] == chrom]
		n_seg = ref['segment'].values[-1] + 1  # num of segments

		ref_seg_med, qry_seg_med, seg_n_bin = [], [], []
		for seg in range(n_seg):  # for each segment
			if seg not in ref['segment'].values:
				continue
			seg_ref = ref.loc[ref['segment'] == seg]['log2_cor_gc_map_repli'].values
			seg_qry = qry.loc[ref['segment'] == seg]['log2_cor_gc_map_repli'].values
			ref_seg_med.append(np.median(seg_ref))
			qry_seg_med.append(np.median(seg_qry))
			seg_n_bin.append(len(seg_ref))

		for seg in range(len(ref_seg_med)):  # for each segment
			if np.std(ref_seg_med) == 0:
				weight = 1
			else:
				weight = 1 + abs(ref_seg_med[seg]) * np.std(qry_seg_med) / np.std(ref_seg_med)

			ref_val = np.ones(seg_n_bin[seg]) * ref_seg_med[seg]
			qry_val = np.ones(seg_n_bin[seg]) * qry_seg_med[seg]
			w = np.ones(seg_n_bin[seg]) * weight

			if ref_bin is None:
				ref_bin = ref_val
				qry_bin = qry_val
				weights = w
			else:
				ref_bin = np.concatenate((ref_bin, ref_val))
				qry_bin = np.concatenate((qry_bin, qry_val))
				weights = np.concatenate((weights, w))

	calibrate = True if re_calibrate and df_qry['depth'].sum()/40/10**7 <= 1 else False
	print('# Query CN re-calibrated')
	frac = nn_wls_lasso(
		ref_bin.reshape(-1, 1),
		qry_bin,
		weights,
		L1_penalty,
		calibrate=calibrate,
		)
	return frac


def matrix_comparison(df_ref, df_qry, similarity_score, output_file):
	coors, vals_qry, vals_ref, meds_ref, meds_qry, seg_lens, bin_nums =\
		[], [], [], [], [], [], []
	for chrom in sorted(set(df_ref['chromosome'].values)):
		if not chrom[3:].isdigit():  # autosome
			continue
		ref = df_ref[df_ref['chromosome'] == chrom]
		qry = df_qry[df_qry['chromosome'] == chrom]
		n_seg = ref['segment'].values[-1] + 1  # num of segments
		for seg in range(n_seg):
			seg_ref = ref.loc[ref['segment'] == seg]['log2_cor_gc_map_repli']
			seg_qry = qry.loc[ref['segment'] == seg]['log2_cor_gc_map_repli']

			if len(ref.loc[ref['segment'] == seg]) == 0:  # filtered chrom
				continue

			start = ref.loc[ref['segment'] == seg]['start'].values[0]
			end = ref.loc[ref['segment'] == seg]['start'].values[-1]
			meds_ref.append(np.median(seg_ref))
			meds_qry.append(np.median(seg_qry))
			coors.append(f'{chrom}:{start/10**6+1:.0f}M')
			seg_lens.append(end-start)
			bin_nums.append(len(seg_ref))
			vals_qry.append(seg_qry)
			vals_ref.append(seg_ref)

	# sort by ref copy number
	srt_idx = np.argsort(meds_ref)
	meds_ref = np.array(meds_ref)[srt_idx]
	meds_qry = np.array(meds_qry)[srt_idx]
	coors = np.array(coors)[srt_idx]
	vals_qry, vals_ref =\
		[vals_qry[i] for i in srt_idx], [vals_ref[i] for i in srt_idx]
	seg_lens = np.array(seg_lens)[srt_idx]

	# pairwise tests
	matrix_qry = scikit_posthocs.posthoc_dunn(vals_qry, p_adjust='sidak').values

	# matrix_qry p-value
	up = matrix_qry[:matrix_qry.shape[0]//2, :matrix_qry.shape[0]//2]
	low = matrix_qry[matrix_qry.shape[0]-matrix_qry.shape[0]//2:,
		matrix_qry.shape[0]-matrix_qry.shape[0]//2:]
	mid = matrix_qry[:matrix_qry.shape[0]//2, -matrix_qry.shape[0]//2:]

	up = up[np.triu_indices(up.shape[0], k=1)].ravel()
	low = low[np.triu_indices(low.shape[0], k=1)].ravel()
	mid = np.delete(mid.ravel(), mid.shape[0]-1)

	# Mann-Whitney U
	_, up_p = stats.mannwhitneyu(mid, up, alternative='greater')
	_, low_p = stats.mannwhitneyu(mid, low, alternative='greater')
	_, ul_p = stats.mannwhitneyu(up, low)

	# Kendall rank correlation coefficient
	tau_seg, tau_p_seg = stats.kendalltau(meds_ref, meds_qry)

	with pdf.PdfPages(output_file[:-4]+'.pdf') as pdf_all:
		leg_strs = []
		for nam, val, p_strs, ps in (
			('Similarity score:', f'{similarity_score:.4f}', (None,), (None,)),
			(r'Kendall’s $\tau$', f'= {tau_seg:.3f}' if abs(tau_seg) < 0.01 else f'= {tau_seg:.2f}', ('$P$',), (tau_p_seg,)),
			('Mann-Whitney $U$ test', '', ('\n  $P_{U}$', ' $P_{L}$', '\n  $P_{U-L}$'), (up_p, low_p, ul_p)),
			):

			leg_str = f'{nam} {val}' if nam is not None else ''
			for p_str, p in zip(p_strs, ps):
				if p is None:
					leg_str += ''
				elif p < 0.001:
					leg_str += f' {p_str} < 0.001'
				elif p < 0.01:
					leg_str += f' {p_str} = {p:.3f}'
				else:
					leg_str += f' {p_str} = {p:.2f}'
			leg_strs.append(leg_str)
		leg_strs.append('\n')
		leg_strs.append(r'P-value (-$log_{10}$)')

		fig = plot_matrix(
			matrix_qry,
			coors,
			seg_lens,
			meds_ref,
			meds_qry,
			leg_strs,
			)
		pdf_all.savefig(fig)

	return tau_seg, tau_p_seg, np.median(up), np.median(mid), np.median(low), up_p, low_p, ul_p


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='cmp.py',
		description='Copy number similarity and significance analyses.')
	parser.add_argument('ref_seg_file')
	parser.add_argument('qry_seg_file')
	parser.add_argument('output_file')
	parser.add_argument('--calibrate', help='If re-calibrate query sample copy number.', action=argparse.BooleanOptionalAction)
	parser.set_defaults(calibrate=True)
	args = parser.parse_args()
	if args.calibrate:
		print('->Calibration')
	else:
		print('->No calibration')

	df_ref, df_qry = load_data(args.ref_seg_file, args.qry_seg_file)
	similarity_score = segment_similarity(df_ref, df_qry, args.calibrate)
	results = matrix_comparison(df_ref, df_qry, similarity_score, args.output_file)

	# output
	with open(args.output_file, 'w') as out:
		out.write(f'{similarity_score}\t')
		res = '\t'.join([f'{r}' for r in results])
		out.write(f'{res}\n')
