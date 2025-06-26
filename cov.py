#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhou Ze'
__version__ = '2.0'

'''GC content, mappability and coverage.
'''

import sys
import gzip
import pysam
import argparse
import pyBigWig
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BIN_SIZE = 100 * 10**3
STEP_SIZE = int(BIN_SIZE/10)

MAPQ = 30


def get_gc_map_cov(arg):
	'''Extract GC content, mappability from bigwig file.
	Get coverage from BAM file.
	'''
	chrom, mappable_regions, gc_file, mappability_file, align_file, minSize, maxSize = arg
	with pysam.AlignmentFile(align_file, 'rb') as bam_file, \
			pyBigWig.open(gc_file) as gc_bw, \
			pyBigWig.open(mappability_file) as map_bw:

		# chromosomal length and init array with zeros
		tot_len = 0
		for start, (end, repli_time) in sorted(mappable_regions.items()):
			tot_len += end - start
		gc_map_repli_pos_dep = np.zeros((5, tot_len), dtype=np.float32)

		# add coordinates and repliation timing to array
		idx = 0
		for start, (end, repli_time) in sorted(mappable_regions.items()):
			# GC and mappability
			gc = gc_bw.values(chrom, start, end)
			mappability = map_bw.values(chrom, start, end)
			gc_map_repli_pos_dep[0, idx: idx+end-start] = gc
			gc_map_repli_pos_dep[1, idx: idx+end-start] = mappability
			gc_map_repli_pos_dep[2, idx: idx+end-start] = repli_time if repli_time is not None else np.nan
			gc_map_repli_pos_dep[3, idx: idx+end-start] = np.arange(start, end)

			# Depth
			for read in bam_file.fetch(chrom, start, end):
				if read.is_unmapped or \
						read.mate_is_unmapped:
					continue
				if read.is_duplicate or \
						read.is_secondary or \
						read.is_supplementary or \
						read.mapping_quality < MAPQ or \
						(not read.is_proper_pair):
					continue
				size = abs(read.template_length)
				if not minSize <= size <= maxSize:
					continue
				if not start <= read.reference_start < end:
					continue
				# read count
				gc_map_repli_pos_dep[4, idx+read.reference_start-start] += 1

			# index update
			idx += end-start

	# summary
	result = []
	for idx, start in enumerate(range(0, tot_len, STEP_SIZE)):
		end = start + BIN_SIZE
		if end > tot_len:
			continue
		gc, mapp, repli, coor, depth = \
			gc_map_repli_pos_dep[:, start: end]

		# depth filtering for each site within 100k
		up_lim_dep = np.percentile(depth, 99)  # 99% upper limit
		# in low depth, aviod trim too many
		if len(np.where(depth <= up_lim_dep)) > 0.98*len(depth):
			gc = gc[np.where(depth <= up_lim_dep)]
			mapp = mapp[np.where(depth <= up_lim_dep)]
			repli = repli[np.where(depth <= up_lim_dep)]
			depth = depth[np.where(depth <= up_lim_dep)]
		tot_dep = np.sum(depth)

		# mean
		gc_mean, map_mean, copy_ratio, repli_mean = \
			np.nanmean(gc), \
			np.nanmean(mapp), \
			np.log2(max(1, np.sum(depth))/BIN_SIZE), \
			np.nanmean(repli)
		result.append(f'{chrom}\t{idx}\t{coor[0]:.0f}\t{coor[-1]:.0f}\t{gc_mean}\t{map_mean}\t{repli_mean}\t{copy_ratio}\t{tot_dep:.0f}\n')
	return result


def main(mappable_file, gc_file, mappability_file, align_file, output_file, thread, minSize, maxSize):
	mappable_regions = {}
	with gzip.open(mappable_file, 'rt') as f:
		for line in f:
			chrom, start, end, repli_time = line.rstrip().split()
			mappable_regions.setdefault(chrom, {})
			mappable_regions[chrom].setdefault(int(start), (int(end), float(repli_time) if repli_time != 'None' else None))

	paramters = [
		(chrom, mappable_regions[chrom], gc_file, mappability_file, align_file, minSize, maxSize)
		for chrom in mappable_regions.keys()]

	with ProcessPoolExecutor(max_workers=thread) as executor:
		results = executor.map(get_gc_map_cov, paramters)

	with gzip.open(output_file, 'wt') as out:
		out.write('chromosome\tbin\tstart\tend\tGC\tMap\tRepli\tlog2\tdepth\n')
		for result in results:
			for line in result:
				out.write(line)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
	prog='cov.py',
	description='Count BAM file read coverage.')
	parser.add_argument('mappable_file')
	parser.add_argument('gc_file')
	parser.add_argument('mappability_file')
	parser.add_argument('align_file')
	parser.add_argument('output_file')
	parser.add_argument('-t', '--thread', help='Number of thread(s) (default 4)',  type=int)
	parser.add_argument('-m', '--minSize', help='Min fragment size (bp; default 30)',  type=int)
	parser.add_argument('-M', '--maxSize', help='Max fragment size (bp; default 1000)',  type=int)
	parser.set_defaults(thread=4)
	parser.set_defaults(minSize=30)
	parser.set_defaults(maxSize=1000)
	args = parser.parse_args()

	main(args.mappable_file, args.gc_file, args.mappability_file, args.align_file, args.output_file, args.thread, args.minSize, args.maxSize)
