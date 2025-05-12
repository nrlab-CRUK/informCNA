#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhou Ze'
__version__ = '2.0'

'''Copy number log2 ratio normalization and segmentation.
Robust B-Spline regression for normalization.
Linearly penalized segmentation (Pelt) for segmentation.
'''

import argparse
import sys
import numpy as np
import pandas as pd

import ruptures as rpt

import scipy.interpolate as si
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.linear_model import HuberRegressor, LinearRegression

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

try:
	sys.path.append('/home/zhou01/python_module')
	import plot_graph
	plot_graph.font_Arial()
except ImportError:
	pass

BIN_SIZE = 100 * 10**3  # same as cov.py
STEP_SIZE = int(BIN_SIZE/10)  # same as cov.py

MIN_N_BINS = int(2*10**6/STEP_SIZE)  # min number of bin in 1 segment ~2Mb


class BSplineFeatures(TransformerMixin):
	def __init__(self, knots, degree=3, periodic=False):
		self.bsplines = get_bspline_basis(knots, degree, periodic=periodic)
		self.nsplines = len(self.bsplines)

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		nsamples, nfeatures = X.shape
		features = np.zeros((nsamples, nfeatures * self.nsplines))
		for ispline, spline in enumerate(self.bsplines):
			istart = ispline * nfeatures
			iend = (ispline + 1) * nfeatures
			features[:, istart:iend] = si.splev(X, spline)
		return features


def get_bspline_basis(knots, degree=3, periodic=False):
	'''Get spline coefficients for each basis spline.'''
	nknots = len(knots)
	y_dummy = np.zeros(nknots)
	knots, coeffs, degree = \
		si.splrep(
				knots,
				y_dummy,
				k=degree,
				per=periodic)
	ncoeffs = len(coeffs)
	bsplines = []
	for ispline in range(nknots):
		coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
		bsplines.append((knots, coeffs, degree))
	return bsplines


def correction(xmin, xmax, X, y, n_knots=10, degree=3, epsilon=1.35):
	'''Correction using spline'''
	knots = np.linspace(xmin, xmax, n_knots)
	bspline_features = BSplineFeatures(knots, degree=degree, periodic=False)

	model = make_pipeline(
			bspline_features,
			LinearRegression())

	# 1st round
	model_1_round = model.fit(X, y)
	y_1_round = model_1_round.predict(X)
	y_1 = y - y_1_round

	model = make_pipeline(
			bspline_features,
			HuberRegressor(
			max_iter=500,
			alpha=1,
			epsilon=1000))

	# 2st round
	kde = gaussian_kde(y_1)
	y_min, y_max = np.quantile(y_1, [0, 0.95])
	y_grid = np.linspace(y_min, y_max, 1000)
	density = kde(y_grid)
	peaks, _ = find_peaks(density)
	peak_densities = density[peaks]
	peak_points = y_grid[peaks]
	max_density_idx = np.argmax(peak_densities)
	#if len(peaks) > 2 and max_density_idx == 0:  # aviod edge
	#	max_density_idx = np.argmax(peak_densities[1:])
	#elif len(peaks) > 2 and max_density_idx == len(peak_densities)-1:
	#	max_density_idx = np.argmax(peak_densities[:-1])
	y_cen = peak_points[max_density_idx]
	if len(peaks) > 1:  # CNA
		y_low, y_up = y_cen-0.5, y_cen+0.5
		for point in peak_points:
			if point > y_cen:
				y_up = min((point+y_cen)/2, y_up)
			elif point < y_cen:
				y_low = max((point+y_cen)/2 , y_low)
		s_X = X[(y_low < y_1) & (y_1 < y_up)]
		s_y = y[(y_low < y_1) & (y_1 < y_up)]
		#print(peak_points, y_cen, peak_densities)
		#print(y_low, '--', y_up, len(s_y)/len(y))
		model.fit(s_X, s_y)
	else:
		model.fit(X, y)
	return model


def plot_correction(df, gc_model, map_model, repli_model, output_file):
	'''Plot before and after correction.'''
	for typ, model, before_column, after_column, xlabel in (
		('GC', gc_model, 'log2', 'log2_cor_gc', 'GC content (%)'),
		('Map', map_model, 'log2_cor_gc', 'log2_cor_gc_map', 'Mappability (%)'),
		('Repli', repli_model, 'log2_cor_gc_map', 'log2_cor_gc_map_repli', 'Replication timing'),
			):
		val = df[typ].values
		fig = plt.figure(figsize=(8, 6))
		plt.plot(df[typ].values,
				df[before_column].values,
				'.',
				color='b',
				alpha=0.99,
				label='Before')
		xmin, xmax = min(val), max(val)
		plt.plot(val,
				df[after_column].values,
				'.',
				color='r',
				alpha=0.1,
				label='After')
		plt.plot(np.linspace(xmin, xmax, 1000),
				model.predict(np.linspace(xmin, xmax, 1000)[:, None]),
				'--',
				lw=2,
				color='k',
				label='Regression')
		plt.xlabel(xlabel, fontsize=20)
		plt.ylabel('Copy ratio (log2)', fontsize=20)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.legend(fontsize=15)
		plt.tight_layout()
		fig.savefig(output_file[:-7] + f'.{typ.lower()}.png')


def plot_genome(df, chr_bkps, chr_typ, output_file, values2=None):
	'''Plot genome copy ratio'''
	width_ratio = [int(df[df['chromosome'] == f'chr{i}']['start'].values[-1]/10**6) for i in range(1, 23)]
	fig = plt.figure(figsize=(10, 40))
	gs = fig.add_gridspec(22, width_ratio[0], hspace=0.5, left=0.1, right=0.95, bottom=0.02, top=0.99)
	axs = [fig.add_subplot(gs[i, 0:w]) for i, w in enumerate(width_ratio)]

	for i in range(22):
		chrom = f'chr{i+1}'
		values = df[df['chromosome'] == chrom]['log2_cor_gc_map_repli'].values
		coor = df[df['chromosome'] == chrom]['start'].values
		prev_b = 0
		for b, typ in zip(chr_bkps[chrom], chr_typ[chrom]):
			val = values[prev_b: b]
			median = np.median(val)
			if typ == 'gain':
				color = 'red'
			elif typ == 'loss':
				color = 'green'
			else:
				color = 'blue'
			print(f'->{chrom}:{coor[prev_b]}-{coor[b-1]}\t{len(val)} bins\t{(coor[b-1]-coor[prev_b])/10**6:.1f} Mb\t{typ}')
			axs[i].scatter(
				coor[prev_b: b]/10**6,
				values[prev_b: b],
				alpha=0.2,
				facecolors=color,
				edgecolors='none',
				s=1,
				)
			axs[i].plot(
				(coor[prev_b]/10**6, coor[b-1]/10**6),
				(median, median),
				ls='-',
				clip_on=False,
				color=color)
			prev_b = b
			if b != chr_bkps[chrom][-1]:
				axs[i].axvline(
					coor[b-1]/10**6,
					color='k',
					ls='--')

		if values2 is not None:
			plt.scatter(
				coor/10**6,
				values2/100*2,
				color='orange',
				s=1,
				)

		axs[i].axhline(0, color='k', ls='-', lw=0.8)
		log = '$log_{2}$'
		axs[i].set_ylabel(f'Copy ratio ({log})')
		axs[i].set_title(f'{chrom} {len(chr_bkps[chrom])} segment(s)', fontsize=12)
		axs[i].set_yticks(
			(-2, -1, 0, 1, 2),
			('-2', '-1', '0', '1', '2'),)
		axs[i].set_ylim(-2, 2)
		axs[i].set_xlim(0, coor[-1]/10**6)

	fig.supxlabel('Coordinate (Mb)', fontsize=15)
	fig.savefig(output_file[:-7]+'.png')


def determine_seg_type(df, segments_coor, segments_median, segments_len, segments_mean, tot_bkps):
	segments_len, segments_median = np.array(segments_len), np.array(segments_median)
	if args.high:  # high tumor fraction
		seg_info = np.column_stack((segments_median, segments_mean))
		cluster = DBSCAN(eps=0.03, min_samples=2).fit_predict(seg_info)
	else:  # low tumor fraction
		print(segments_median)
		seg_info = np.reshape(segments_median, (-1, 1))
		cluster = HDBSCAN(min_samples=2).fit_predict(seg_info)
	if len(set(cluster)) == 1:  # -1 outliers
		neutral_cluster = cluster[0]
		neutral_median = 0
	else:
		cluster_idx = list(range(max(cluster)))
		cluster_len = [np.sum(segments_len[cluster == i]) for i in cluster_idx]
		neutral_idx = np.argmax(cluster_len)
		cluster_median = [np.median(segments_median[cluster == i]) for i in cluster_idx]
		neutral_cluster = cluster_idx[neutral_idx]
		neutral_median = cluster_median[neutral_idx]

	chr_typ = {}
	for (chrom, pos), val, cls, (p_bkp, bkp) in \
			zip(segments_coor, segments_median, cluster, tot_bkps):
		if cls == neutral_cluster and neutral_cluster != -1:
			typ = 'neutral'
		elif val < neutral_median:
			typ = 'loss'
		elif val >= neutral_median:
			typ = 'gain'
		else:
			raise ValueError()
		chr_typ.setdefault(chrom, [])
		chr_typ[chrom].append(typ)
	return chr_typ


def get_neutral_median(df, chr_bkps, chr_typ):
	neutral_bins = None
	for chrom in sorted(set(df['chromosome'].values)):
			if not chrom[3:].isdigit():  # autosome
				continue
			values = df[df['chromosome'] == chrom]['log2_cor_gc_map_repli'].values
			prev_b = 0
			for b, typ in zip(chr_bkps[chrom], chr_typ[chrom]):
				val = values[prev_b: b]
				if typ == 'neutral':
					neutral_bins = val if neutral_bins is None else np.concatenate((neutral_bins, val))
				prev_b = b
	return np.median(neutral_bins)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='seg.py',
		description='Copy number segmentation.')
	parser.add_argument('copy_number_file')
	parser.add_argument('output_file')
	#parser.add_argument('--pon', help='Panal of normals txt.gz file')
	parser.add_argument('--high', help='High tumour fraction sample', action=argparse.BooleanOptionalAction)
	parser.set_defaults(high=False)
	parser.set_defaults(pon=False)
	args = parser.parse_args()
	if args.high:
		print('->High tumour fraction mode')
	else:
		print('->Generic mode')

	# load copy number
	df = pd.read_csv(
		args.copy_number_file,
		sep='\t',
		header=0,
		compression='gzip').dropna()
	ori_len = len(df)
	df = df[df['Map'] > 0.9]
	df = df[df['GC'] <= 60]

	df['segment'] = 0
	print(f'Keep {len(df)/ori_len*100:.1f}% bins')

	# GC content correction
	auto_df = df.loc[df['chromosome'].str.contains(r'\d$')]
	gc = auto_df['GC'].values
	y = auto_df['log2'].values
	GC = gc[:, np.newaxis]
	xmin, xmax = min(df['GC'].values), max(df['GC'].values)
	x = np.linspace(xmin, xmax, 1000)
	# fit
	gc_model = correction(xmin, xmax, GC, y, n_knots=4, degree=3, epsilon=3)
	# predict
	y = df['log2'].values
	gc = df['GC'].values
	y_predicted = gc_model.predict(gc[:, None])
	df['log2_cor_gc'] = y - y_predicted
	print('Finish GC content correction')

	# mappability correction
	auto_df = df.loc[df['chromosome'].str.contains(r'\d$')]
	mapp = auto_df['Map'].values
	y = auto_df['log2_cor_gc']
	xmin, xmax = min(df['Map'].values), max(df['Map'].values)
	MAPP = mapp[:, np.newaxis]
	# fit
	map_model = correction(xmin, xmax, MAPP, y, n_knots=3, degree=2, epsilon=10)
	# predict
	mapp = df['Map'].values
	y = df['log2_cor_gc']
	y_predicted = map_model.predict(mapp[:, None])
	df['log2_cor_gc_map'] = y - y_predicted
	print('Finish mappability correction')

	# replication timing correction
	auto_df = df.loc[df['chromosome'].str.contains(r'\d$')]
	repli = auto_df['Repli'].values
	y = auto_df['log2_cor_gc_map']
	xmin, xmax = min(df['Repli'].values), max(df['Repli'].values)
	REPLI = repli[:, np.newaxis]
	# fit
	repli_model = correction(xmin, xmax, REPLI, y, n_knots=3, degree=2, epsilon=10)
	# predict
	repli = df['Repli'].values
	y = df['log2_cor_gc_map']
	xmin, xmax = min(repli), max(repli)
	y_predicted = repli_model.predict(repli[:, None])
	df['log2_cor_gc_map_repli'] = y - y_predicted
	print('Finish replication timing correction')

	# panel of normals correction
	if args.pon:  # PoN file exists
		pon_df = pd.read_csv(
			args.pon,
			sep='\t',
			header=0,
			compression='gzip').dropna()
		df['coor'] = df[['chromosome', 'start', 'end']].apply(tuple, axis=1)
		pon_df['coor'] = pon_df[['chromosome', 'start', 'end']].apply(tuple, axis=1)
		shared_coor = df[df['coor'].isin(pon_df['coor'])]['coor']
		df = df.loc[df['coor'].isin(shared_coor)].reset_index()
		pon_df = pon_df.loc[pon_df['coor'].isin(shared_coor)].reset_index()
		df['log2_cor_gc_map_repli'] -= pon_df['PoN'] if 'PoN' in pon_df else pon_df['log2_cor_gc_map_repli']
		df['log2_cor_gc_map_repli'] -= np.median(df.loc[df['chromosome'].str.contains(r'\d$')]['log2_cor_gc_map_repli'])

	plot_correction(df, gc_model, map_model, repli_model, args.output_file)

	# segmentation
	segments_median, segments_mean, segments_coor, segments_len, tot_bkps, chr_bkps = [], [], [], [], [], {}
	for chrom in sorted(set(df['chromosome'].values)):
		signal = df[df['chromosome'] == chrom]['log2_cor_gc_map_repli'].values
		coor = df[df['chromosome'] == chrom]['start'].values
		std = np.std(signal)
		pen_bic = 10 * np.log(len(signal))*std if args.high else 5 * np.log(len(signal))*std
		# Linearly penalized segmentation (Pelt) l2
		algo = rpt.KernelCPD(
				kernel='linear',
				min_size=MIN_N_BINS,
				).fit(signal)

		try:
			bkps = algo.predict(pen=pen_bic)
			chr_bkps[chrom] = bkps
			cost = algo.cost.sum_of_costs(bkps)
			print(f'Finish {chrom} segmentaion n={len(bkps)}')
		except rpt.exceptions.BadSegmentationParameters:
			print(f'Excluded {chrom}')
			continue

		row_index = df.loc[df['chromosome'] == chrom].index
		p_bkp = 0
		for i, bkp in enumerate(bkps):  # add boundary
			df.loc[row_index[p_bkp: bkp], 'segment'] = i
			segments_coor.append((chrom, df.loc[row_index[p_bkp], 'start']))
			segments_median.append(np.median(df.loc[row_index[p_bkp: bkp], 'log2_cor_gc_map_repli'].values))
			segments_mean.append(np.mean(df.loc[row_index[p_bkp: bkp], 'log2_cor_gc_map_repli'].values))
			tot_bkps.append((p_bkp, bkp))
			segments_len.append(bkp-p_bkp)
			p_bkp = bkp

	# determine gain/loss/neutral
	chr_typ = determine_seg_type(df, segments_coor, segments_median, segments_len, segments_mean, tot_bkps)

	if args.high:  # recalibration for high tumour fraction sample
		df['log2_cor_gc_map_repli'] -= get_neutral_median(df, chr_bkps, chr_typ)

	# plot genome copy number per chromosome
	plot_genome(
		df,
		chr_bkps,
		chr_typ,
		args.output_file,
		)

	# output
	df.to_csv(args.output_file, sep='\t', compression='gzip', index=False)
