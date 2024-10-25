# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:50:06 2023

# uorfish  

v0.9
_x_ fix final bed output
_x_ output cutoff
_x_ fix load_psite_object


v0.14
_x_ added 40s sub calling
_x_ incremented batch n = 100 from 10

v1.0 (ejovalip)
_x_ public beta

v1.1 (boaconic):
    #metadata release
        _x_ incorporate https://www.sciencedirect.com/science/article/pii/S2405471220302404
    
    #code
        _x_ change validated_uorfs input to gff
        _x_ add p_site_toggle
        
    ___ one line command
        _x_ ui text for stage completion

v1.2 (saz)
    #metadata release
        _x_ tested for non-start TIS in gff
        _x_ corrected ~150 coordinate misannotations

v1.3 (canitingers)
    _x_ improved peak distribution 
    _x_ highqc variable

future versions
    ___ qc predictions co-ordinates
    ___ prefilter predictions for near-AUG
    ___ one line command        
    ___ clean up load_psite_object
    ___ add multiple p_site determinations
    ___ add RNA input
    ___ multiple initialzation ramp feature types - eg 3, 4, 5 (10.1038/s41467-019-13810-1), not just 4
        
###
Design notes:

    #validated uorfs (The left nt is -1 from what IGV displays)

"""

'''
'xx' = unevaluated
'TT' = True positive | True and in training data
'Tf' = False positive | True but not in training data
'Ft' = False negative | False and in training data
'Ff' = True negative | False and not in training data

if features_dict[uid]['fdr'] <= cut_off:
    if features_dict[uid]['_is_uorf']:
        cm['Tt'] += 1 
        features_dict[uid]['cm']='Tt'
    else:
        cm['Tf'] += 1
        features_dict[uid]['cm']='Tf'
else:
    if features_dict[uid]['_is_uorf']:
        cm['Ft'] += 1 
        features_dict[uid]['cm']='Ft'
    else:
        cm['Ff'] += 1
        features_dict[uid]['cm']='Ff'

'''

"""

python uorfish.py -name DGY1657_rep1 -config
python uorfish.py -name DGY1657_rep1 -genome
python uorfish.py -name DGY1657_rep1 -read -sam C:/Gresham/Project_Carolino/MBE_Review_materials/sam/RPF_DGY1657_R1.sorted.sam
python uorfish.py -name DGY1657_rep1 -features -known
python uorfish.py -name DGY1657_rep1 -train
python uorfish.py -name DGY1657_rep1 -predict

@author: Pieter Spealman, pspealman@nyu.edu
"""
# for base
from pathlib import Path
import numpy as np
import pandas as pd
import re
import argparse
import pickle
import random
import subprocess

# for psite_analysis
import plotly.io as pio
pio.renderers.default = "browser"

import plotly.graph_objects as go

# for torch
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

psite_bedgraph_name_list = []

#io
parser = argparse.ArgumentParser()

parser.add_argument('-run', '--run_analysis', default = False, action = "store_true")

parser.add_argument('-config', '--configure_run', default = False, action = "store_true")
parser.add_argument('-name', '--run_name')
parser.add_argument('-o',"--output")
parser.add_argument('-i',"--input")

parser.add_argument('-genome',"--build_genome_object", default = False, action = "store_true")
parser.add_argument('-tl', '--transcript_leader_filename')
parser.add_argument('-fa', '--fasta_reference_file')
parser.add_argument('-dubious', '--dubious_orfs_filename')
parser.add_argument('-go',"--genome_object")
parser.add_argument('-go_gff', '--genome_gff_object')

parser.add_argument('-load_g',"--load_genome_object", default = False, action = "store_true")

parser.add_argument('-read',"--parse_read_object", default = False, action = "store_true")
parser.add_argument('-sam',"--sam_filename")
parser.add_argument('-bedgraph',"--make_bedgraph", default = True, action = "store_true")
parser.add_argument('-ro',"--read_object")

parser.add_argument('-known',"--parse_uorf_object", default = False, action = "store_true")
parser.add_argument('-uvt',"--uorf_validation_type")
parser.add_argument('-val_uorfs',"--validated_uorfs_filename")
parser.add_argument('-highqc',"--highqc_candidate_cutoff", default = 0.05)
parser.add_argument('-psite_mode',"--psite_mode", default = "singular")


parser.add_argument('-features',"--build_features_object", default = False, action = "store_true")
parser.add_argument('-fo',"--features_object_filename")
parser.add_argument('-to',"--train_object_features")
parser.add_argument('-po',"--predict_object_features")

parser.add_argument('-train',"--train_model", default = False, action = "store_true")
parser.add_argument('-model', "--dnn_model_filename")

parser.add_argument('-predict',"--predict", default = False, action = "store_true")
parser.add_argument('-cutoff',"--prediction_cutoff")
args = parser.parse_args()

'''
# General functions
'''
start_codons = set(['ATG', 'TTG', 'GTG', 'CTG', 'ATA', 'ATT', 'ACG', 'ATC'])

stop_codons = set(['TAA', 'TAG', 'TGA'])

sign_to_text = {'+':'plus','-':'minus'}

def reverse_complement(sequence):
    sequence = sequence.upper()
    complement = sequence.translate(str.maketrans('ATCG', 'TAGC'))
    return(complement[::-1])

def write_resource_object(resource_object, resources_dict):
    
    resource_file = open(resource_object, 'w')
    
    for metadata, value in resources_dict.items():
        outline = ('{metadata}\t{value}\n').format(metadata = metadata, value = value)
        resource_file.write(outline)
    resource_file.close()
    
def parse_resource_object():
    
    run_name = args.run_name
    
    if (args.run_analysis):
        return()
    
    if args.configure_run:
        Path(run_name).mkdir(parents=True, exist_ok=True)
    
    resource_filename = ('{}/_resources.tab').format(run_name)
    file_path = Path(resource_filename)
    
    if file_path.is_file():
        resources_dict = {}

        resource_file = open(resource_filename)
        for line in resource_file:
            line = line.strip()
            #print(line)
            metadata, value = line.split('\t')
            resources_dict[metadata] = value
        resource_file.close()
        
        resources_dict['resource_filename'] = resource_filename
                       
        if args.transcript_leader_filename:
            resources_dict['transcript_leader_filename'] = args.transcript_leader_filename
                    
        if args.fasta_reference_file:
            resources_dict['fasta_reference_file'] = args.fasta_reference_file
        
        if args.dubious_orfs_filename:
            resources_dict['dubious_orfs_filename'] = args.dubious_orfs_filename
            
        if args.genome_object:
            resources_dict['genome_object'] = args.genome_object
        
        if args.genome_gff_object:
            resources_dict['genome_gff_object'] = args.genome_gff_object
            
        if args.uorf_validation_type:
            resources_dict['uorf_validation_type'] = args.uorf_validation_type
            
        if args.validated_uorfs_filename:
            resources_dict['validated_uorfs_filename'] = args.validated_uorfs_filename
        
        if args.features_object_filename:
            resources_dict['features_object_filename'] = args.features_object_filename
            
        if args.sam_filename:
            resources_dict['sam_filename'] = args.sam_filename
            
        if args.read_object:
            resources_dict['read_object'] = args.read_object
            
        if args.make_bedgraph:
            resources_dict['make_bedgraph'] = args.make_bedgraph
            
        if args.parse_uorf_object:
            resources_dict['parse_uorf_object'] = args.parse_uorf_object

        if args.train_object_features:
            resources_dict['train_object_features'] = args.train_object_features
            
        if args.predict_object_features:
            resources_dict['predict_object_features'] = args.predict_object_features
            
        if args.dnn_model_filename:
            resources_dict['dnn_model_filename'] = args.dnn_model_filename
            
        if args.output:
            resources_dict['output'] = args.output
            
        if args.prediction_cutoff:
            resources_dict['prediction_cutoff'] = args.prediction_cutoff
                            
    else:
        if (not args.configure_run):
            outline = ('Missing configuration file for run name: {}\n exiting now ...')
            print(outline)
            exit()
            
        if args.configure_run:
            resources_dict = {}
            
            bedgraph_filenames = ('{run_name}/{run_name}_bedgraph_filenames.p').format(run_name = run_name)
            resources_dict['bedgraph_filenames'] = bedgraph_filenames
            
            if args.transcript_leader_filename:
                resources_dict['transcript_leader_filename'] = args.transcript_leader_filename
            else:
                resources_dict['transcript_leader_filename'] = 'demo/saccharomyces_cerevisiae_transcript_leader.gff'
                        
            if args.fasta_reference_file:
                resources_dict['fasta_reference_file'] = args.fasta_reference_file
            else:
                resources_dict['fasta_reference_file'] = 'demo/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa'
                
            if args.dubious_orfs_filename:
                resources_dict['dubious_orfs_filename'] = args.dubious_orfs_filename
            else:
                resources_dict['dubious_orfs_filename'] = 'demo/dubious_orfs.tsv'
                                        
            if args.uorf_validation_type:
                resources_dict['uorf_validation_type'] = args.uorf_validation_type
            else:
                resources_dict['uorf_validation_type'] = 'molecular'
                    
            if args.validated_uorfs_filename:
                resources_dict['validated_uorfs_filename'] = args.validated_uorfs_filename
            else:
                #resources_dict['validated_uorfs_filename'] = 'demo/validated_uorfs.csv'
                resources_dict['validated_uorfs_filename'] = 'demo/validated_uorfs_v1.2_test.gff'
                
            if args.genome_object:
                resources_dict['genome_object'] = args.genome_object
            else:
                outline = ('{run_name}/_genome_object.p').format(run_name = run_name)
                resources_dict['genome_object'] = outline
                
            if args.genome_gff_object:
                resources_dict['genome_gff_object'] = args.genome_gff_object
            else:
                outline = ('{run_name}/_genome_gff_object.p').format(run_name = run_name)
                resources_dict['genome_gff_object'] = outline
                
            if args.features_object_filename:
                resources_dict['features_object_filename'] = args.features_object_filename
            else:
                outline = ('{run_name}/_features_object.csv').format(run_name = run_name)
                resources_dict['features_object_filename'] = outline
                    
            if args.sam_filename:
                resources_dict['sam_filename'] = args.sam_filename
            else:
                outline = ('USER_SPECIFIED')
                resources_dict['sam_filename'] = outline
                    
            if args.read_object:
                resources_dict['read_object'] = args.read_object
            else:
                outline = ('{run_name}/_read_object.p').format(run_name = run_name)
                resources_dict['read_object'] = outline
                
            if args.make_bedgraph:
                resources_dict['make_bedgraph'] = args.make_bedgraph
            else:
                resources_dict['make_bedgraph'] = False
                
            if args.parse_uorf_object:
                resources_dict['parse_uorf_object'] = args.parse_uorf_object
            else:
                resources_dict['parse_uorf_object'] = True
                
            if args.train_object_features:
                resources_dict['train_object_features'] = args.train_object_features
            else:
                outline = ('{run_name}/_train_object_features.csv').format(run_name = run_name)
                resources_dict['train_object_features'] = outline
                
            if args.predict_object_features:
                resources_dict['predict_object_features'] = args.predict_object_features
            else:
                outline = ('{run_name}/_predict_object_features.csv').format(run_name = run_name)
                resources_dict['predict_object_features'] = outline
                
            if args.prediction_cutoff:
                resources_dict['prediction_cutoff'] = args.prediction_cutoff
            else:
                resources_dict['prediction_cutoff'] = 0.0
                
            if args.dnn_model_filename:
                resources_dict['dnn_model_filename'] = args.dnn_model_filename
            else:
                outline = ('{run_name}/_dnn_model.p').format(run_name = run_name)
                resources_dict['dnn_model_filename'] = outline                
                
            if args.output:
                resources_dict['output'] = args.output
            else:
                outline = ('{run_name}/{run_name}').format(run_name = run_name)
                resources_dict['output'] = outline
                
            if args.psite_mode:
                resources_dict['psite_mode'] = args.psite_mode
                
            if args.highqc_candidate_cutoff:
                resources_dict['highqc_candidate_cutoff'] = args.highqc_candidate_cutoff
    
    write_resource_object(resource_filename, resources_dict)
    
    return(resources_dict)

'''
Read in potential TL coordinates to get nucleotide sequences
'''

def return_dubious_orfs():
    dubious_orfs_filename = resources_dict['dubious_orfs_filename']
    dubious_orfs = set()
    
    infile = open(dubious_orfs_filename)
    
    for line in infile:
        line = line.strip()
        gene, qualifier = line.split('\t')
        if qualifier == 'Dubious':
            dubious_orfs.add(gene)
            
    return(dubious_orfs)
            
def make_TL_fasta():
    transcript_leader_filename = resources_dict['transcript_leader_filename']
    fasta_reference_file = resources_dict['fasta_reference_file']
    
    coord_dict = {}
    
    dubious_orf = return_dubious_orfs()
        
    tl_file = open(transcript_leader_filename)
    
    for line in tl_file:
        line = line.strip()
        
        if 'five_prime_UTR' in line:
            chromo = line.split('\t')[0]
            
            if 'chr' in chromo:
                chromo = chromo.split('chr')[1]
            
            start = int(line.split('\t')[3])
            stop =int(line.split('\t')[4])
            sign = line.split('\t')[6]
            
            gene = line.split('\t')[8].split('=')[1].split('_')[0]
            
            if gene[0] == 'Y' and gene not in dubious_orf:
                if gene not in coord_dict:
                    coord_dict[gene] = {}
                    
                coord_dict[gene] = {'chromo':chromo,'start':start, 'stop':stop, 'sign': sign}
                        
    tl_file.close()
                    
    '''
    read in chromosomes
    
    '''
    fa_dict = {}
    fa_file = open(fasta_reference_file)
    
    for line in fa_file:
        if line[0] != '#':
            line = line.strip()
        
            if line[0] == '>':
                chromo = line.split('>')[1].split(' ')[0]
                if chromo not in fa_dict:
                    fa_dict[chromo] = ''
                        
            else:
                fa_dict[chromo]+=line
            
    fa_file.close()  
    
    return(fa_dict, coord_dict)

'''
Let's find the start codons
'''
def get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, isit):
    
    if coord_sign == '+':
        if isit == 'start':
            nt = coord_start + codon_start + 1
        else:
            nt = coord_start + codon_stop
        
    if coord_sign == '-':
        if isit == 'start':
            nt = coord_stop - codon_start
        else:
            nt = coord_stop - codon_stop + 1
            
    return(nt)

def find_puorf_regions(gene, puorf_dict, fa_dict, coord_dict):
    
    print('Finding putative uORFs in gene: ', gene)
    seq, coord_chromo, coord_start, coord_stop, coord_sign = get_sequences(gene, fa_dict, coord_dict)
    
    if 'CHR' in coord_chromo.upper():
        
        coord_chromo = coord_chromo.upper().replace('CHR','')

    set_of_start_codons = set()
                 
    for frame in set([0,1,2]):
        #print('frame', frame)   
        previous_hit_count = -1
        hit_count = 0
        
        while hit_count != previous_hit_count:
            #print('hitcount', hit_count, previous_hit_count)
            
            previous_hit_count = hit_count
            
            in_puorf = False
            
            # Instead of juggling cuts and frames I just ask if a codon is produced and evaluate on that
            for i in range(len(seq)):
                codon_start = frame+0+i*3
                codon_stop = frame+3+i*3
                 
                codon = seq[codon_start:codon_stop]
                
                if codon:
                    #print(i, codon_start, codon)
                    
                    if not in_puorf:
                        if codon in start_codons:
                            #print('start')
                            if codon_start not in set_of_start_codons:
                                set_of_start_codons.add(codon_start)
                                
                                in_puorf = True
                                
                                start_codon = codon
                                puorf_start = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'start')
                    else:
                        if codon in stop_codons:
                            in_puorf = False
        
                            stop_codon = codon
                            puorf_stop = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'stop')           
                            
                            if gene not in puorf_dict:
                                puorf_dict[gene]={}
                                
                            puorf_dict[gene][len(puorf_dict[gene])] = {
                                'puorf_start': puorf_start,
                                'puorf_stop': puorf_stop,
                                'start_codon': start_codon,
                                'stop_codon': stop_codon
                                }
                            
                            hit_count +=1
                            
            if in_puorf: #handles oORFs
                stop_codon = 'oORF'
                #puorf_stop = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'stop')           
                
                if gene not in puorf_dict:
                    puorf_dict[gene]={}
                
                if coord_sign == '+':
                    puorf_dict[gene][len(puorf_dict[gene])] = {
                        'puorf_start': puorf_start,
                        'puorf_stop': coord_stop - 1,
                        'start_codon': start_codon,
                        'stop_codon': stop_codon
                        }
                else:
                    puorf_dict[gene][len(puorf_dict[gene])] = {
                        'puorf_start': puorf_start,
                        'puorf_stop': coord_start,
                        'start_codon': start_codon,
                        'stop_codon': stop_codon
                        }
                    
                hit_count +=1
                

    #here we're collecting the information for the whole TL
    if gene in puorf_dict:
        if len(puorf_dict[gene]) > 0:
            puorf_dict[gene][-1] = {
                'chromo':coord_chromo,
                'coord_start':coord_start,
                'coord_stop':coord_stop,
                'coord_sign':coord_sign
                }
        
    return(puorf_dict)

def make_puorf_scramble(fa_dict, coord_dict, relative_puorf_dict, start_codon_counter):
    puorf_scramble_dict = {}
    
    for gene in coord_dict:
        if gene in relative_puorf_dict:
            psite_dict = relative_puorf_dict[gene][-1]['psite_dict']
            mask_stop = relative_puorf_dict[gene][-1]['mask_stop']

            unscrambled_seq, coord_chromo, coord_start, coord_stop, coord_sign = get_sequences(gene, fa_dict, coord_dict)
            
            for frame in set([0,1,2]):
                unwound_seq = []
                
                for i in range(len(unscrambled_seq)):
                    codon_start = frame+0+i*3
                    codon_stop = frame+3+i*3
                     
                    codon = unscrambled_seq[codon_start:codon_stop]

                    if len(codon) == 3:
                        unwound_seq.append(codon)
                        
                for sim in range(3):
                    seq_list = random.sample(unwound_seq, len(unwound_seq))
                    
                    seq = ''
                    for each in seq_list:
                        seq += each
                        
                    set_of_start_codons = set()
                                 
                    for frame in set([0,1,2]):
                        #print('frame', frame)   
                        previous_hit_count = -1
                        hit_count = 0
                        
                        while hit_count != previous_hit_count:
                            #print('hitcount', hit_count, previous_hit_count)
                            
                            previous_hit_count = hit_count
                            in_puorf = False
                            
                            # Instead of juggling cuts and frames I just ask if a codon is produced and evaluate on that
                            for i in range(len(seq)//3):
                                codon_start = frame+0+i*3
                                codon_stop = frame+3+i*3

                                codon = seq[codon_start:codon_stop]
                                psite = psite_dict[codon_start]
                                
                                if codon and (psite > 1):
                                    
                                    if not in_puorf:
                                        if codon in start_codons:
                                            if codon_start not in set_of_start_codons:
                                                set_of_start_codons.add(codon_start)
                                                
                                                in_puorf = True
                                                
                                                start_codon = codon
                                                puorf_start = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'start')
                                                # start_codon_counter[codon]+=1
                                                
                                    else:
                                        if codon in stop_codons:
                                            in_puorf = False
                        
                                            stop_codon = codon
                                            puorf_stop = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'stop')           
                                            
                                            if gene not in puorf_scramble_dict:
                                                puorf_scramble_dict[gene]={}
                                                
                                            puorf_scramble_dict[gene][len(puorf_scramble_dict[gene])] = {
                                                'puorf_start': puorf_start,
                                                'puorf_stop': puorf_stop,
                                                'start_codon': start_codon,
                                                'stop_codon': stop_codon,
                                                'left': min(puorf_start, puorf_stop),
                                                'right': max(puorf_start, puorf_stop),
                                                }
                                            
                                            hit_count +=1
                                            
                            if in_puorf: #handles oORFs
                                stop_codon = 'oORF'
                                #puorf_stop = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'stop')           
                                
                                if gene not in puorf_scramble_dict:
                                    puorf_scramble_dict[gene]={}
                                
                                if coord_sign == '+':
                                    puorf_scramble_dict[gene][len(puorf_scramble_dict[gene])] = {
                                        'puorf_start': puorf_start,
                                        'puorf_stop': coord_stop - 1,
                                        'start_codon': start_codon,
                                        'stop_codon': stop_codon,
                                        'left': puorf_start,
                                        'right': coord_stop -1
                                        }
                                else:
                                    puorf_scramble_dict[gene][len(puorf_scramble_dict[gene])] = {
                                        'puorf_start': puorf_start,
                                        'puorf_stop': coord_start,
                                        'start_codon': start_codon,
                                        'stop_codon': stop_codon,
                                        'left': coord_start,
                                        'right': puorf_start
                                        }
                                    
                                hit_count +=1
                        
                #here we're collecting the information for the whole TL
                if gene in puorf_scramble_dict:
                    if len(puorf_scramble_dict[gene]) > 0:
                        puorf_scramble_dict[gene][-1] = {
                            'chromo':coord_chromo,
                            'coord_start':coord_start,
                            'coord_stop':coord_stop,
                            'coord_sign':coord_sign,
                            'mask_stop':mask_stop,
                            'psite_dict': psite_dict                            
                            }
                    
    relative_puorf_scramble_dict = strand_agnostic(puorf_scramble_dict, psite_object)
    puorf_scramble_features = {}
    
    relative_puorf_scramble_dict, puorf_scramble_features = build_features(puorf_scramble_features, relative_puorf_scramble_dict, set())
            
    for uid in puorf_scramble_features:
        if puorf_scramble_features[uid]['pass_filter_one']:
            is_lump = puorf_scramble_features[uid]['lump']
            start_codon = puorf_scramble_features[uid]['start_codon']
            rs = puorf_scramble_features[uid]['resolved_score']
                    
            start_codon_counter[is_lump][start_codon].append(rs)
            
    return(start_codon_counter)

# def select_balanced_gene(positive_genes, count_cut_off):
    
#     gene_list = list(positive_genes.keys())
#     ct = np.inf
#     loop_ct = len(gene_list)

#     while (ct >= count_cut_off) and (loop_ct > 0):
#         select_gene_list = random.sample(gene_list, k = 1)
#         select_gene = select_gene_list[0]
#         ct = positive_genes[select_gene]
        
#         loop_ct -= 1 
           
#     return(select_gene)

# def make_synthetic_false_puorf(fa_dict, coord_dict, cut_off_lookup_dict, relative_puorf_dict, puorf_features):
#     fdr_cut_off = 0.05 
    
#     # select genes pass QC and count the number of 'Ff' (True false) puorfs that fail QC
#     positive_genes = {}
#     negative_puorfs = set()
    
#     for uid in puorf_features:
#         if (puorf_features[uid]['fdr'] < fdr_cut_off) and (puorf_features[uid]['cm'] == 'Tt'):
#             gene = puorf_features[uid]['_gene']
            
#             if gene not in positive_genes:
#                 positive_genes[gene] = 0
                            
#         if (puorf_features[uid]['fdr'] >= fdr_cut_off) and (puorf_features[uid]['cm'] == 'Ff'):
#             puorf_features[uid]['cm'] = 'filtered_Ff'
#             negative_puorfs.add(uid)    
    
#     print('positive_genes', len(positive_genes))
#     print('negative_puorfs', len(negative_puorfs))
    
#     puorf_scramble_dict = {}
#     for uid in negative_puorfs:
        
#         gene = select_balanced_gene(positive_genes, 2)
#         positive_genes[gene] += 1 
        
#         #print('select_gene', gene)
        
#         if gene in relative_puorf_dict:
#             psite_dict = relative_puorf_dict[gene][-1]['psite_dict']
#             mask_stop = relative_puorf_dict[gene][-1]['mask_stop']

#             unscrambled_seq, coord_chromo, coord_start, coord_stop, coord_sign = get_sequences(gene, fa_dict, coord_dict)
            
#             for frame in set([0,1,2]):
#                 unwound_seq = []
                
#                 for i in range(len(unscrambled_seq)):
#                     codon_start = frame+0+i*3
#                     codon_stop = frame+3+i*3
                     
#                     codon = unscrambled_seq[codon_start:codon_stop]

#                     if len(codon) == 3:
#                         unwound_seq.append(codon)
                        
#                 for sim in range(3):
#                     seq_list = random.sample(unwound_seq, len(unwound_seq))
                    
#                     seq = ''
#                     for each in seq_list:
#                         seq += each
                        
#                     set_of_start_codons = set()
                                 
#                     for frame in set([0,1,2]):
#                         #print('frame', frame)   
#                         previous_hit_count = -1
#                         hit_count = 0
                        
#                         while hit_count != previous_hit_count:
#                             #print('hitcount', hit_count, previous_hit_count)
                            
#                             previous_hit_count = hit_count
#                             in_puorf = False
                            
#                             # Instead of juggling cuts and frames I just ask if a codon is produced and evaluate on that
#                             for i in range(len(seq)//3):
#                                 codon_start = frame+0+i*3
#                                 codon_stop = frame+3+i*3

#                                 codon = seq[codon_start:codon_stop]
#                                 psite = psite_dict[codon_start]
                                
#                                 if codon and (psite > 1):
                                    
#                                     if not in_puorf:
#                                         if codon in start_codons:
#                                             if codon_start not in set_of_start_codons:
#                                                 set_of_start_codons.add(codon_start)
                                                
#                                                 in_puorf = True
                                                
#                                                 start_codon = codon
#                                                 puorf_start = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'start')
#                                                 # start_codon_counter[codon]+=1
                                                
#                                     else:
#                                         if codon in stop_codons:
#                                             in_puorf = False
                        
#                                             stop_codon = codon
#                                             puorf_stop = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'stop')           
                                            
#                                             if gene not in puorf_scramble_dict:
#                                                 puorf_scramble_dict[gene]={}
                                                
#                                             puorf_scramble_dict[gene][len(puorf_scramble_dict[gene])] = {
#                                                 'puorf_start': puorf_start,
#                                                 'puorf_stop': puorf_stop,
#                                                 'start_codon': start_codon,
#                                                 'stop_codon': stop_codon,
#                                                 }
                                            
#                                             hit_count +=1
                                            
#                             if in_puorf: #handles oORFs
#                                 stop_codon = 'oORF'
#                                 #puorf_stop = get_igv_coords(coord_sign, coord_start, coord_stop, codon_start, codon_stop, 'stop')           
                                
#                                 if gene not in puorf_scramble_dict:
#                                     puorf_scramble_dict[gene]={}
                                
#                                 if coord_sign == '+':
#                                     puorf_scramble_dict[gene][len(puorf_scramble_dict[gene])] = {
#                                         'puorf_start': puorf_start,
#                                         'puorf_stop': coord_stop - 1,
#                                         'start_codon': start_codon,
#                                         'stop_codon': stop_codon,
#                                         }
#                                 else:
#                                     puorf_scramble_dict[gene][len(puorf_scramble_dict[gene])] = {
#                                         'puorf_start': puorf_start,
#                                         'puorf_stop': coord_start,
#                                         'start_codon': start_codon,
#                                         'stop_codon': stop_codon,
#                                         }
                                    
#                                 hit_count +=1
                        
#                 #here we're collecting the information for the whole TL
#                 if gene in puorf_scramble_dict:
#                     if len(puorf_scramble_dict[gene]) > 0:
#                         puorf_scramble_dict[gene][-1] = {
#                             'chromo':coord_chromo,
#                             'coord_start':coord_start,
#                             'coord_stop':coord_stop,
#                             'coord_sign':coord_sign,
#                             'mask_stop':mask_stop,
#                             'psite_dict': psite_dict                            
#                             }
                    
#     relative_puorf_scramble_dict = strand_agnostic(puorf_scramble_dict, psite_object)
#     puorf_scramble_features = {}
    
#     print('relative_puorf_scramble_dict', relative_puorf_scramble_dict.keys())
    
#     relative_puorf_scramble_dict, puorf_scramble_features = build_features(puorf_scramble_features, relative_puorf_scramble_dict, set())
            
#     qc_puorf_scramble_features = {}
    
#     for scramble_uid in puorf_scramble_features:
#         islump = puorf_scramble_features[scramble_uid]['lump']
#         start_codon = puorf_scramble_features[scramble_uid]['start_codon']
#         rs = puorf_scramble_features[scramble_uid]['resolved_score']
        
#         cut_off = cut_off_lookup_dict[islump][start_codon]
        
#         if rs >= cut_off:
#             qc_puorf_scramble_features[scramble_uid] = puorf_scramble_features[scramble_uid]
#             qc_puorf_scramble_features[scramble_uid]['fdr'] = fdr_cut_off
#             qc_puorf_scramble_features[scramble_uid]['cm'] = 'Sf'
            
            
#     scramble_uid_list = list(qc_puorf_scramble_features.keys())
    
#     random.shuffle(scramble_uid_list)
    
#     for i in range(len(negative_puorfs)):
#         uid = len(puorf_features)
        
#         scramble_uid = scramble_uid_list[i]
#         puorf_features[uid] = puorf_scramble_features[scramble_uid]
        
#         print(puorf_features[uid])
            
#     return(puorf_features)
                
def get_sequences(gene, fa_dict, coord_dict):
    coord_chromo = coord_dict[gene]['chromo']
    coord_start = coord_dict[gene]['start']
    coord_stop = coord_dict[gene]['stop']
    coord_sign = coord_dict[gene]['sign']
    
    if coord_chromo in fa_dict:
        coord_seq = fa_dict[coord_chromo][coord_start:coord_stop]
        
    if coord_sign == '-':
        coord_seq = reverse_complement(coord_seq)
        
    return(coord_seq, coord_chromo, coord_start, coord_stop, coord_sign)

def make_gff(puorf_dict):
    genome_gff_object = resources_dict['genome_gff_object']
    gff_file = open(genome_gff_object, 'w')
    
    for gene in puorf_dict:
        
        sign = puorf_dict[gene][-1]['coord_sign']
        chromo = puorf_dict[gene][-1]['chromo']
            
        for puorf in puorf_dict[gene]:
            if puorf >= 0:
                '''
                'chromo':coord_chromo,
                'puorf_start':puorf_start,
                'puorf_stop':puorf_stop,
                'sign':coord_sign,
                'start_codon':start_codon,
                'stop_codon':stop_codon
                '''
                
                if sign == '+':
                    left = puorf_dict[gene][puorf]['puorf_start']
                    right = puorf_dict[gene][puorf]['puorf_stop']
                    name = ('{}.{}').format(gene,left)
                    
                else:
                    left = puorf_dict[gene][puorf]['puorf_stop']
                    right = puorf_dict[gene][puorf]['puorf_start']
                    name = ('{}.{}').format(gene,right)
                    
                #chrI	AWN	mRNA	136914	137510	.	+	.	ID=YAL008W_mRNA;PARENT=YAL008W
                outline = ('{chromo}\tcue_orf\tuORF\t{left}\t{right}'
                           '\t.\t{sign}\t.\tID={name}_uORF;Parent={gene}\n').format(
                               chromo = chromo, left = left, right = right,
                               sign = sign, name = name, gene = gene)
                gff_file.write(outline)
            
    gff_file.close()         

def demarcate_regions(fa_dict, coord_dict):
    puorf_dict = {}
    
    dubious_orfs = return_dubious_orfs()
    
    for gene in coord_dict:
        if gene not in dubious_orfs:
            puorf_dict = find_puorf_regions(gene, puorf_dict, fa_dict, coord_dict)
            
    make_gff(puorf_dict)
        
    return(puorf_dict)

def test_start_codon_saturation(start_codon_counter):
    #print(start_codon_counter)
    status = False
    
    cutoff = 1e4
    print('Testing codon type for null model, cut off is: ', cutoff)
    
    for mode in start_codon_counter:
        print(mode)
        for codon in start_codon_counter[mode]:
            print(codon, len(start_codon_counter[mode][codon]))
            
            #print(codon, start_codon_counter[mode][codon])
        
            if len(start_codon_counter[mode][codon]) < cutoff:
                #print('test_start_codon_saturation', codon, len(start_codon_counter[mode][codon]))
                status = True
                        
    return(status)


def calculate_scramble_pval(fa_dict, coord_dict):
    
    outline = ('Generating random distribution for null model ...')
    print(outline)
    
    start_codon_set = start_codons
    
    start_codon_counter = {}
    for mode in set(['lump', 'peak']):
        start_codon_counter[mode] = {}
        
        for start_codon in start_codon_set:
            start_codon_counter[mode][start_codon] = []
            
    while test_start_codon_saturation(start_codon_counter):
        start_codon_counter = make_puorf_scramble(fa_dict, coord_dict, relative_puorf_dict, start_codon_counter)
                
    return(start_codon_counter)
        
# def make_bedgraph(psite_dict, runsize, match_sign):
#     sign = sign_to_text[match_sign]
    
#     outfile_name = ('{run_name}/{run_name}_{runsize}_{sign}.bedgraph').format(
#         run_name = args.run_name,
#         runsize = runsize,
#         sign = sign)
#     outfile = open(outfile_name, 'w')
    
#     print('making bedgraph: ', outfile_name)
    
#     for chromo in psite_dict:
#         min_nt = min(psite_dict[chromo])
#         max_nt = max(psite_dict[chromo])
        
#         for nt in range(min_nt, max_nt+1):
#             if nt in psite_dict[chromo]:
#                 outline = ('{chromo}\t{start}\t{stop}\t{score}\n').format(
#                     chromo = chromo,
#                     start = nt-1,
#                     stop = nt,
#                     score = psite_dict[chromo][nt])
#                 #print(outline)
#                 outfile.write(outline)
                
#     outfile.close()
    
#     return(outfile_name)

# def add_to_psite_dict(psite_dict, chromo, coord, weight):    
#     if chromo not in psite_dict:
#         psite_dict[chromo] = {}
        
#     if coord not in psite_dict[chromo]:
#         psite_dict[chromo][coord] = 0
        
#     psite_dict[chromo][coord] += weight
    
#     return(psite_dict)

# def qual_seq(sequence, poly_n=8):
#     sequence = sequence.upper()
#     for each_nt in ['A', 'T', 'G', 'C']:
#         search_str = each_nt * poly_n
#         if search_str not in sequence:
#             return(True)
    
#     return(False)    

def unpackbits(x,num_bits=12):
    '''
    SAM flags are bit encoded - this function seperates them and assigns labels
    '''
    
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    upb = (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

    #0  (rp)    read_paired
    #1  (rmp)    read_mapped_in_proper_pair
    #2  (ru)    read_unmapped
    #3  (mu)    mate_unmapped
    #4  (rrs)    read_reverse_strand
    #5  (mrs)    mate_reverse_strand
    #6  (fip)    first_in_pair
    #7  (sip)    second_in_pair
    #8  (npa)    not_primary_alignment
    #9  (rfp)    read_fails_platform
    #10 (pcr)    read_is_PCR_or_optical_duplicate
    #11 (sa)    supplementary_alignment
    """
    strand = unpackbits(np.array([int(line.split('\t')[1])]))[0][4]
    such that 0 == '+' and 1 == '-'
    """
    
    """ DISCORDANT definition (from samblaster)
        Both side of the read pair are mapped (neither FLAG 0x4 or 0x8 is set).
        The properly paired FLAG (0x2) is not set.
        Note: We implemented an additional criteria to distinguish between strand re-orientations and distance issues
        Strand Discordant reads must be both on the same strand.
    """
        
    """ SPLIT READS
        Identify reads that have between two and --maxSplitCount [2] primary and supplemental alignments.
        Sort these alignments by their strand-normalized position along the read.
        Two alignments are output as splitters if they are adjacent on the read, and meet these criteria:
            each covers at least --minNonOverlap [20] base pairs of the read that the other does not.
            the two alignments map to different reference sequences and/or strands. 
            the two alignments map to the same sequence and strand, and represent a SV that is at least --minIndelSize [50] in length, 
            and have at most --maxUnmappedBases [50] of un-aligned base pairs between them.
        Split read alignments that are part of a duplicate read will be output unless the -e option is used.
    """
    
    return(upb) 

# def parse_cigar(cigar, sequence):
#     """This function calculates the offset for the read based on the match
#         returns an sequence uncorrected to sign
#     """

#     if cigar.count('M') == 1:
#         #print(cigar, sequence)
#         left_cut = 0
#         right_cut = 0
        
#         left_list = re.split('M|S|D|I|H|N', cigar.split('M')[0])[0:-1]
#         M = re.split('M|S|D|I|H|N', cigar.split('M')[0])[-1]
#         right_list = re.split('M|S|D|I|H|N', cigar.split('M')[1])
        
#         for each in left_list:
#             if each: 
#                 left_cut += int(each)
                
#         for each in right_list:
#             if each: 
#                 right_cut -= int(each)
        
#         n_cigar = ('{}M').format(M)

#         if right_cut:
#             n_sequence = sequence[left_cut:right_cut]
#         else:
#             n_sequence = sequence[left_cut:]
                        
#         #print (left_cut, right_cut,  n_cigar, n_sequence)
#         return(True, n_cigar, n_sequence)
            
#     else:
#         return(False, '', '')
    
# TODO make this a callable option
# def make_asite_fraction(asite_dict, n_sequence, chromo, start, stop, sign, size):
#     #from https://www.nature.com/articles/s41598-019-42348-x
    
#     asite_frac_lookup = {24:set([15]),
#                          25:set([15]),
#                          26:set([15,12]),
#                          27:set([15]),
#                          28:set([15]),
#                          29:set([15]),
#                          30:set([15]),
#                          31:set([15]),
#                          32:set([18,15]),
#                          33:set([18]),
#                          34:set([18])}
    
#     size = len(n_sequence)
#     if size in asite_frac_lookup:
#         fraction_set = asite_frac_lookup[size]
        
#         weight = round(1/(len(fraction_set)),2)
        
#         for offset in fraction_set:
#             if sign == "+":
#                 zero_site_coord = start + offset
                
#             if sign == "-":
#                 zero_site_coord = stop - offset
                
#             coord = zero_site_coord
        
#             if chromo not in asite_dict:
#                 asite_dict[chromo] = {}
                
#             if coord not in asite_dict[chromo]:
#                 asite_dict[chromo][coord] = 0
                
#             asite_dict[chromo][coord] += weight
            
#             return(asite_dict)

# def make_psite_fraction(psite_dict, n_sequence, chromo, start, stop, sign, size):

#     size = len(n_sequence)
#     offset_bins = abs(size - 28)
            
#     if sign == "+":
#         zero_site_coord = start + 12
        
#     if sign == "-":
#         zero_site_coord = stop - 13
        
#     weight = round(1/(1+(2*offset_bins)),5)
    
#     add_to_psite_dict(psite_dict, chromo, zero_site_coord, weight)
        
#     for each in range(offset_bins):
#         offset = each + 1
#         mod_site_coord = zero_site_coord + offset 
        
#         add_to_psite_dict(psite_dict, chromo, mod_site_coord, weight)
        
#         mod_site_coord = zero_site_coord - offset 
        
#         add_to_psite_dict(psite_dict, chromo, mod_site_coord, weight)
        
#     return(psite_dict)

# def parse_rpf_sam_file(samfile_name, minsize = 20, maxsize = np.inf, match_sign = '+', reverse=False):
#     psite_dict = {}
#     line_ct = 0
        
#     '''
#     open sam file and populate read dictionary, save as pickle
#     '''
#     #@PG	
#     #@CO
#     #SRR10302098.18096714	0	I	24007	1	1M	*	0	0	A	*	NH:i:3	HI:i:1	AS:i:29	nM:i:0
#     #SRR10302098.11813844	16	I	24022	1	1M	*	0	0	G	*	NH:i:3	HI:i:1	AS:i:29	nM:i:0
#     #SRR10302098.19932359	16	I	24022	1	1M	*	0	0	G	*	NH:i:3	HI:i:1	AS:i:29	nM:i:0
        
#     strand_to_sign = {0:'+', 1:'-'}
#     if reverse:
#         strand_to_sign = {0:'-', 1:'+'}
    
#     samfile = open(samfile_name)
#     print('Processing RPF sam file ...', samfile_name)
    
#     #adjusted_sam_file_name = samfile_name.split('.sam')[0] + '_psite_adj.sam'
#     #adjusted_sam_file = open(adjusted_sam_file_name, 'w')

#     for line in samfile:
#         if (line_ct >= 1e6) and (line_ct % 1e6) == 0:
#             print('Processing ', line_ct)
#         # if line[0] == '@':
#         #     adjusted_sam_file.write(line)
                    
#         if line[0] != '@':
#             line_ct += 1
            
#             chromo = line.split('\t')[2]
                                    
#             mapq = int(line.split('\t')[4])
#             cigar = line.split('\t')[5]
#             sequence = line.split('\t')[9]

#             ### short read rna and rpf 
#             process, n_cigar, n_sequence = parse_cigar(cigar, sequence)
            
#             if (process) and (mapq >= 1):
#                 sign = strand_to_sign[unpackbits(np.array([int(line.split('\t')[1])]))[0][4]]

#                 start = int(line.split('\t')[3])
#                 size = len(n_sequence)
#                 stop = start + size
             
#                 if size >= minsize and size <= maxsize and sign == match_sign:
#                     if qual_seq(n_sequence):
#                        psite_dict = make_psite_fraction(psite_dict, n_sequence, chromo, start, stop, sign, size)
                            
#     return(psite_dict)

'''
parse_rpf
'''
# def store_bedgraph(size, match_sign, bedgraph_dict, psite_dict):
#     generate_bedgraph = resources_dict['make_bedgraph']
    
#     if generate_bedgraph:
#         outfile_name = make_bedgraph(psite_dict, size, match_sign)
        
#         if size not in bedgraph_dict[match_sign]:
#             bedgraph_dict[match_sign][size] = set()
            
#         bedgraph_dict[match_sign][size].add(outfile_name)
        
#     return(bedgraph_dict)

def parse_psite_cigar(cigar):
    '''
    short read rna and rpf can be generated using library prep techniques that
    add/ligate bases that are exogenous to the fragment. To recover a better 
    estimation of the true fragment size we can remove bases that have been added.
    '''
                
    """This function calculates the offset for the read based on the match
        For the 'first_match' run_mode it always takes the furthest left M, as this is the assigned start in sam file format
        if additional nucleotide types preceed the match these need to be stripped out. 
        For the 'last_match' run_mode it always takes the fullest length from the alignement.
        Such that it includes each M and each deletion 'D', it does not include insertions, hard or soft clips. 
    """

    #left_cut, right_cut, match_length
    '''
    cigar = '1H2I4D' return (0)
    cigar = '4M' return (4)
    cigar = '1H2I4M8D16M32N64H' for this: 
        #NB - since this is to remove library added bases we are not targeting the removal
        of mid sequence mismatches - therefore we want to trim the ends.
        1. we want to take the 1H2I as the left cut.
        2. we want to take the 32N64H as the right cut.
        3. we want to return the entire middle as the match, regardless of the call
        return (28)
    '''
    
    if cigar.count('M') == 0:
        return(0)
    
    else:
        left_cut = 0
        mismatch_list = re.split('M|S|D|I|H|N', cigar.split('M')[0])[0:-1]
        for mismatch in mismatch_list:
            if mismatch:
                left_cut += int(mismatch)
        
        right_cut = 0 
        mismatch_list = re.split('M|S|D|I|H|N', cigar.rsplit('M',1)[1])
        for mismatch in mismatch_list:
            if mismatch:
                right_cut += int(mismatch)
        
        match_length = 0
        mismatch_list = re.split('M|S|D|I|H|N', cigar)
        for mismatch in mismatch_list:
            if mismatch:
                match_length += int(mismatch)    
        match_length -= (left_cut + right_cut)
        
        return(match_length)
    
    #unexpected input
    return(0)

# get CDS TIS
def parse_gff_for_cds_tis(genome_gff_name):
    
    tis_dict = {}
    
    gff_file = open(genome_gff_name)
    
    tis_ct = 0
    
    for line in gff_file:
        if line[0] != '#':
            #I	sgd	CDS	335	649	.	+	0	ID=CDS:YAL069W;Parent=transcript:YAL069W_mRNA;protein_id=YAL069W
            region_type = line.split('\t')[2].lower()
            exon_count = line.split('\t')[7]
    
            if (region_type == 'cds') and (exon_count == '0'):
                chromo = line.split('\t')[0]
                
                if chromo not in tis_dict:
                    tis_dict[chromo] = {'+':set(),
                                        '-':set()}
                                    
                sign = line.split('\t')[6]
                
                if sign == '+':
                    tis = int(line.split('\t')[3])+1
    
                if sign == '-':
                    tis = int(line.split('\t')[4])-1
                    
                tis_dict[chromo][sign].add(tis)
                
                tis_ct+=1
                                   
    print(tis_ct)
    
    return(tis_dict)

def populate_p5_site(size, start, stop, chromo, sign, tis_dict, p5_site_dict, 
                     p5_site_dict_plus, p5_site_dict_minus):
    
    if chromo in tis_dict:
        if sign in tis_dict[chromo]:
            for tis in tis_dict[chromo][sign]:
                if tis >= start and tis <= stop:

                    if sign == '+':
                        tis_distance = start-tis
                        
                        if tis_distance not in p5_site_dict_plus[size]:
                            p5_site_dict_plus[size][tis_distance] = 0
                            
                        p5_site_dict_plus[size][tis_distance] += 1
                        
                    if sign == '-':
                        tis_distance = -1*(stop - tis - 1)

                        if tis_distance not in p5_site_dict_minus[size]:
                            p5_site_dict_minus[size][tis_distance] = 0
                            
                        p5_site_dict_minus[size][tis_distance] += 1
                        
                    if tis_distance not in p5_site_dict[size]:
                        p5_site_dict[size][tis_distance] = 0
                        
                    p5_site_dict[size][tis_distance] += 1
    
    return(p5_site_dict, p5_site_dict_plus, p5_site_dict_minus)

def populate_fragment_object(sign, chromo, nt, size, fragment_data_object):
    if sign not in fragment_data_object:
        fragment_data_object[sign] = {}
    
    if chromo not in fragment_data_object[sign]:
        fragment_data_object[sign][chromo] = {}
        
    if nt not in fragment_data_object[sign][chromo]:
        fragment_data_object[sign][chromo][nt] = {}
        
    if size not in fragment_data_object[sign][chromo][nt]:
        fragment_data_object[sign][chromo][nt][size] = 0
        
    fragment_data_object[sign][chromo][nt][size] += 1
    
    return(fragment_data_object)
            
def remove_homopolymers(seq, homopolymer_density_cutoff):
    if len(seq) <= 0:
        return(False)
        
    else:
        temp_seq = seq.upper()
        for nuc in ['A', 'C', 'T', 'G', 'N']:
            homopolymer_density = temp_seq.count(nuc)/max(len(temp_seq),1)
            
            if homopolymer_density > homopolymer_density_cutoff:
                return(False)
                
        return(True)

#identify RPFs that overlap
def select_rpf_overlap(sam_filename, tis_dict,  minsize = 20, maxsize = np.inf):   
    p5_site_dict = {}
    p5_site_dict_plus = {}
    p5_site_dict_minus = {}
    
    fragment_data_object = {}
    
    for size in range(minsize, maxsize+1):
        p5_site_dict[size] = {}
        p5_site_dict_plus[size] = {}
        p5_site_dict_minus[size] = {}
    
    line_ct = 0
    hit_ct = 0
    highm = 0
    
    mapq_distribution = {'Optimal reads: ': 0,
                         'Mapq 0: ': 0,
                         'Outside size range: ': 0,
                         'Multimapper primary: ': 0,
                         'Multimapper secondary: ': 0,
                         'Homopolymer reads (polyN): ': 0}
        
    '''
    open sam file and populate read dictionary, save as pickle
    '''
    #@PG	
    #@CO
    #SRR10302098.18096714	0	I	24007	1	1M	*	0	0	A	*	NH:i:3	HI:i:1	AS:i:29	nM:i:0
    #SRR10302098.11813844	16	I	24022	1	1M	*	0	0	G	*	NH:i:3	HI:i:1	AS:i:29	nM:i:0
    #SRR10302098.19932359	16	I	24022	1	1M	*	0	0	G	*	NH:i:3	HI:i:1	AS:i:29	nM:i:0
        
    strand_to_sign = {0:'+', 1:'-'}
    
    samfile = open(sam_filename)
    print('Processing RPF sam file ...', sam_filename)
    excluded_samfile_name = ('{run_name}/_{run_name}_excluded_reads.sam').format(
        run_name = run_name)
    
    print('Excluded reads are saved to: ', excluded_samfile_name)
    
    excluded_samfile = open(excluded_samfile_name, 'w')
    
    for line in samfile:
        if (line_ct >= 1e6) and (line_ct % 1e6) == 0:
            print('Processing ', line_ct)
            
        if line[0] == '@':
            excluded_samfile.write(line)
                    
        if line[0] != '@':
            line_ct += 1
            
            '''
            STAR aligner uses mapping quality of 255 for uniquely mapping reads,
            for multimapped reads the best scoring alignment is primary and secondary
            alignments have their sam flag (0x100) set. 
            
            Here any multimapper that is secondary will be removed
            '''
            process = False
            
            mapq = int(line.split('\t')[4])
            
            if (mapq == 255):
                mapq_distribution['Optimal reads: '] += 1 
                process = True
            
            if (mapq != 255):
                if (mapq == 0):
                    mapq_distribution['Mapq 0: '] += 1
                    excluded_samfile.write(line)
                    
                if (mapq > 0):
                    flag = int(line.split('\t')[1])
                    
                    if flag < 256:
                        mapq_distribution['Multimapper primary: '] += 1
                        process = True
                    else:
                        mapq_distribution['Multimapper secondary: '] += 1
                        excluded_samfile.write(line)
            
            seq = line.split('\t')[9]        
            not_homopolymer = remove_homopolymers(seq, 0.75)
            if not not_homopolymer:
                mapq_distribution['Homopolymer reads (polyN): '] += 1
                excluded_samfile.write(line)

            if process and not_homopolymer:                
                #sequence = line.split('\t')[9]

                cigar = line.split('\t')[5]
                if cigar.count('M')>1:
                    highm += 1
                    
                size = parse_psite_cigar(cigar)
                
                # size = len(n_sequence)
                
                # if (size >= minsize) and (size <= maxsize):
                if (size >= minsize) and (size <= maxsize):
                    hit_ct += 1
                    
                    chromo = line.split('\t')[2]
                    if 'chr' in chromo:
                        chromo = chromo.split('chr')[1]    
                    
                    sign = strand_to_sign[unpackbits(np.array([int(line.split('\t')[1])]))[0][4]]
                    
                    '''
                    Because the parse_cigar returned the size minus the unaligned ends 
                    we can adjust the stop appropriately
                    '''
                    start = int(line.split('\t')[3])
                    stop = start + size
                 
                    # build p5_site object
                    p5_site_dict, p5_site_dict_plus, p5_site_dict_minus = populate_p5_site(size, start, stop, chromo, sign, tis_dict, 
                                                                                           p5_site_dict, p5_site_dict_plus, p5_site_dict_minus)
                                    
                    # build fragment object
                    fragment_data_object = populate_fragment_object(sign, chromo, start, size, fragment_data_object)
                    
                else:
                    mapq_distribution['Outside size range: '] += 1
                    excluded_samfile.write(line)
    
    samfile.close()
    excluded_samfile.close()
    
    #TODO make norm_multiplier optional
    norm_multiplier = 1e6/hit_ct
    
    outline = ('Summary:\n'
               '\tTotal hits: {hit_ct} \n'
               '\tNormalization multiplier: {norm_multiplier} \n'
               '\tComplex matching reads recovered: {highm}, {highm_pct}%').format(
                   hit_ct = hit_ct, 
                   norm_multiplier = norm_multiplier,
                   highm = highm,
                   highm_pct = round((100*highm/hit_ct),3))
    print(outline)

    print('Read distribution:')
    for mapq in mapq_distribution:
        print(mapq, mapq_distribution[mapq])
        
    removed = (mapq_distribution['Mapq 0: '] + 
                mapq_distribution['Outside size range: '] + 
                mapq_distribution['Multimapper secondary: '] +
                mapq_distribution['Homopolymer reads (polyN): '])

        
    outline = ('Reads used: {kept}, {kept_pct}%\n'
                'Reads removed: {removed}, {removed_pct}%').format(
                    kept = hit_ct, 
                    kept_pct = (100*hit_ct/line_ct),
                    removed = removed, 
                    removed_pct = (100*removed / line_ct))
                    
    print(outline)
        
    return(p5_site_dict, p5_site_dict_plus, p5_site_dict_minus, fragment_data_object, norm_multiplier)

def make_psite_bedgraph(sample_read_object, runmode, norm_multiplier, bedgraph_set):

    runname = sample_read_object['reference_name']
    data_type = sample_read_object['data_type']
    psite_dict = sample_read_object[runmode]
    
    for sign in psite_dict:
        textsign = sign_to_text[sign]
        
        outfile_name = ('{runname}/{runname}_{data_type}_{runmode}_{textsign}.bedgraph').format(
            runname = runname,
            data_type = data_type,
            runmode = runmode,
            textsign = textsign)
        outfile = open(outfile_name, 'w')
        
        print('making bedgraph: ', outfile_name)
        
        for chromo in psite_dict[sign]:
            min_nt = min(psite_dict[sign][chromo])
            max_nt = max(psite_dict[sign][chromo])
            
            for nt in range(min_nt, max_nt+1):
                if nt in psite_dict[sign][chromo]:
                    outline = ('{chromo}\t{start}\t{stop}\t{score}\n').format(
                        chromo = chromo,
                        start = nt-1,
                        stop = nt,
                        score = psite_dict[sign][chromo][nt]*norm_multiplier)
                    #print(outline)
                    outfile.write(outline)
                    
        outfile.close()
            
        bedgraph_set.add(outfile_name)
    
    return(bedgraph_set)

def make_psite_fractions(fragment_data_object, weight_dict, sample_read_object, runmode = 'singular'): 
    psite_dict = {'+':{}, '-':{}}
           
    #fragment_data_object[sign][chromo][nt][size]
    for sign in fragment_data_object:
        for chromo in fragment_data_object[sign]:
            for start in fragment_data_object[sign][chromo]:
                for size in fragment_data_object[sign][chromo][start]:
                    stop = start + size
                    val = fragment_data_object[sign][chromo][start][size]
                    
                    if runmode == 'singular':
                        offset = abs(weight_dict[size])
                        
                        if sign == '+':
                            psite = start + offset - 1
                                                    
                        if sign == '-':
                            psite = stop - offset
                            
                        if chromo not in psite_dict[sign]:
                            psite_dict[sign][chromo] = {}
                            
                        if psite not in psite_dict[sign][chromo]:
                            psite_dict[sign][chromo][psite] = 0
                            
                        psite_dict[sign][chromo][psite] += val   
                        
                    if runmode == 'multiple':
                        for distance in weight_dict[size]:
                            offset = abs(distance)
                            modifier = weight_dict[size][distance]
                        
                            if sign == '+':
                                psite = start + offset - 1
                                                        
                            if sign == '-':
                                psite = stop - offset
                                
                            if chromo not in psite_dict[sign]:
                                psite_dict[sign][chromo] = {}
                                
                            if psite not in psite_dict[sign][chromo]:
                                psite_dict[sign][chromo][psite] = 0
                                
                            psite_dict[sign][chromo][psite] += val*modifier
        
    if runmode in sample_read_object:
        print('error duplicate data or duplicate names in input files')
        
    else:
        sample_read_object[runmode] = psite_dict
            
    return(sample_read_object)

def make_heatmap(tis_dict, runname, sign):   
    #because of zero indexing the usual literature value of -12 will be shown as -11
    x_set = set()
    y_set = set()
    
    for size in tis_dict:
        y_set.add(size)
        for distance in tis_dict[size]:
            x_set.add(distance)
    
    x = list(x_set)
    y = list(y_set)
    
    x.sort()
    y.sort()
    
    #populate with zeros
    for size in y:
        for distance in x:
            if distance not in tis_dict[size]:
                tis_dict[size][distance] = 0

    z = []
    for size in y:
        sub_z = []
        for distance in x:
            z_val = tis_dict[size][distance]
            # z_val = np.log2(max(tis_dict[size][distance],1))
            sub_z.append(z_val)

        z.append(sub_z)

    
    fig = go.Figure(data=go.Heatmap(
                       z=z,
                       x=x,
                       y=y,
                       hoverongaps = False))
    
    fig_title = ('{runname} {sign}').format(
        runname = runname,
        sign = sign) 
    
    fig.update_layout(
    title=fig_title,
    xaxis_title="Distance from P-site",
    yaxis_title="Fragment size",
    legend_title="Abundance",

    )
    
    fig.show()
    
def make_max_weight(p5_site_dict):
    
    single_point_dict = {}
    
    for size in p5_site_dict:
        optimal_distance_val = 0
        optimal_distance = 0
        
        for distance in p5_site_dict[size]:
            val = p5_site_dict[size][distance]
            
            if val > optimal_distance_val:
                optimal_distance_val = val
                optimal_distance = distance
                
        single_point_dict[size] = optimal_distance
        
    
        
    return(single_point_dict)

def make_split_weight(p5_site_dict):
    
    split_point_dict = {}
    
    for size in p5_site_dict:
        if size not in split_point_dict:
            split_point_dict[size] = {}
        
        total_val = 0
        
        for distance in p5_site_dict[size]:
            val = p5_site_dict[size][distance]
            total_val += val
            
        for distance in p5_site_dict[size]:
            val = p5_site_dict[size][distance]
            pct_val = val/max(total_val,1)
        
            split_point_dict[size][distance] = pct_val
        
    return(split_point_dict)    
            


    
def parse_reads(sample_read_object, sam_filename, bedgraph_set): 
    '''
    size = '28nt'
    #only 28mers
    minsize = 28
    maxsize = 28
    
    if size not in rpf_dict:
        rpf_dict[size] = {'+': {}, '-': {}}
        
    match_sign = '+'
    psite_dict = parse_rpf_sam_file(sam_filename, minsize, maxsize, match_sign)
    rpf_dict[size][match_sign] = psite_dict
    bedgraph_dict = store_bedgraph(size, match_sign, bedgraph_dict, psite_dict)
    
    match_sign = '-'
    psite_dict = parse_rpf_sam_file(sam_filename, minsize, maxsize, match_sign)
    rpf_dict[size][match_sign] = psite_dict
    bedgraph_dict = store_bedgraph(size, match_sign, bedgraph_dict, psite_dict)
    
    size = 'all'
    minsize = 15 #Young 2021
    maxsize = np.inf
    
    if size not in rpf_dict:
        rpf_dict[size] = {'+': {}, '-': {}}
    
    match_sign = '+'
    psite_dict = parse_rpf_sam_file(sam_filename, minsize, maxsize, match_sign)
    rpf_dict[size][match_sign] = psite_dict    
    bedgraph_dict = store_bedgraph(size, match_sign, bedgraph_dict, psite_dict)
    
    match_sign = '-'
    psite_dict = parse_rpf_sam_file(sam_filename, minsize, maxsize, match_sign)
    rpf_dict[size][match_sign] = psite_dict
    bedgraph_dict = store_bedgraph(size, match_sign, bedgraph_dict, psite_dict)
    
    return(rpf_dict, bedgraph_dict)
    
    '''
    
    # TODO make data type CLI
    runname = args.run_name
    psite_mode = resources_dict['psite_mode']
    
    
    # TODO finish data load interface
    # if runname not in sample_read_object:
    #     sample_read_object[runname] = {}
        
    data_type = 'RPF' # 'SSU'/'RNA'/'RPF'
    
    # if data_type not in sample_read_object[runname]:
    #     sample_read_object[runname] = {}
    sample_read_object = {'data_type':data_type,
                        'reference_name': runname}

    # get CDS TIS
    genome_gff_name = ('C:/Gresham/genomes/Ensembl/Saccharomyces_cerevisiae.R64-1-1.50.gff3')
    tis_dict = parse_gff_for_cds_tis(genome_gff_name)
    
    #
    #identify RPFs that overlap
    sam_filename = args.sam_filename
    #
    p5_site_dict, p5_site_dict_plus, p5_site_dict_minus, fragment_data_object, norm_multiplier = select_rpf_overlap(sam_filename, tis_dict, 15, 80)
    
    norm_multiplier = 1
    
    make_heatmap(p5_site_dict, runname, '')
    #make_heatmap(p5_site_dict_plus, runname, '+')
    #make_heatmap(p5_site_dict_minus, runname, '-')
    
    if psite_mode == "singular":
        weight_dict = make_max_weight(p5_site_dict)
        sample_read_object = make_psite_fractions(fragment_data_object, weight_dict, sample_read_object, 'singular')
        bedgraph_set = make_psite_bedgraph(sample_read_object, 'singular', norm_multiplier, bedgraph_set)
        
    
    # psite_dict = make_psite_fractions(sam_filename, single_point_dict, 29, 29, 'singular')
    # make_bedgraph(psite_dict, 'singular_29', runname, norm_multiplier)
    if psite_mode == "multiple":
        weight_dict = make_split_weight(p5_site_dict)
        sample_read_object = make_psite_fractions(fragment_data_object, weight_dict, sample_read_object, 'multiple')
        bedgraph_set = make_psite_bedgraph(sample_read_object, 'multiple', norm_multiplier, bedgraph_set)
        
    return(sample_read_object, bedgraph_set)

'''
# parse uorfs
'''



def parse_uorf():
    uorf_validation_type = resources_dict['uorf_validation_type']
    validated_uorfs_filename = resources_dict['validated_uorfs_filename']
    
    outline = ("... using validated uORFs: {}").format(validated_uorfs_filename)
    print(outline)
    
    validated_uorfs = {}

    '''
    	gene	start	stop	sign	is_uorf	source	validated
    0	YAL019W-A	114222	oORF	+	TRUE	may_2023	molecular

    '''
    
    df = pd.read_csv(validated_uorfs_filename, index_col = 0)
    
    if uorf_validation_type != 'all':
        df = df.loc[df['validated'] == uorf_validation_type]
        
    validated_uorfs = df.to_dict('index')
    
    for vuorf in validated_uorfs:
        if validated_uorfs[vuorf]['start'] != 'oORF':
            validated_uorfs[vuorf]['start'] = int(validated_uorfs[vuorf]['start'])
            
        if validated_uorfs[vuorf]['stop'] != 'oORF':
            validated_uorfs[vuorf]['stop'] = int(validated_uorfs[vuorf]['stop'])
   
    return(validated_uorfs)

def parse_gff_uorf():
    uorf_validation_type = resources_dict['uorf_validation_type']
    validated_uorfs_filename = resources_dict['validated_uorfs_filename']
        
    outline = ("... using validated uORFs: {}").format(validated_uorfs_filename)
    print(outline)
    
    validated_uorfs = {}
    '''
    {'gene': gene,
     'start': uORFstart,
     'stop': uORFend,
     'sign': sign,
     'is_uorf': signif,
     'source': 'may_2023',
     'validated': 'molecular',
     'matched': False}
    '''

    '''
    #v1.2 format
    I	validated_orf	oORF	114221	114259	.	+	.	ID=YAL019W-A_114221_may_2023;Parent=YAL019W-A;Source=may_2023;aTIS=ATG;Stop=TGA;validated=molecular;is_uORF=TRUE
    '''
    validated_uorfs_file = open(validated_uorfs_filename)
    
    for line in validated_uorfs_file:
        if line[0] != '#':
            line = line.strip()
            #print(line)
            chromo, _v, _t, left, right, _d1, sign, _d2, deets = line.split('\t')
            start = int(left)
            stop = int(right)
            
            vuorf = deets.split('ID=')[1].split(';')[0]
            gene = deets.split('Parent=')[1].split(';')[0]
            source = deets.split('Source=')[1].split(';')[0]
            is_uorf = deets.split('is_uORF=')[1].split(';')[0]
            validated = deets.split('validated=')[1].split(';')[0]
            
            if 'T' in is_uorf.upper():
                is_uorf = True
            
            else:
                if 'F' in is_uorf.upper():
                    is_uorf = False
                        
            if (uorf_validation_type == 'all') or (validated == uorf_validation_type):
                if vuorf not in validated_uorfs:
                    validated_uorfs[vuorf] = {'gene': gene,
                         'start': start,
                         'stop': stop,
                         'sign': sign,
                         'is_uorf': is_uorf,
                         'source': source,
                         'validated': validated,
                         'matched': False}
       
    return(validated_uorfs)

'''
# build_features_object
'''
def load_psite_object(psite_bedgraph_name, psite_object):

    print('Running load_psite_object...', psite_bedgraph_name)
    
    # TODO revist loading sam files instead, bedgraphs lack the strand sensitivity that is needed 
    # text_and_sign = {'plus':'+', 
    #                    'minus':'-'}
    
    sign = ''
    if 'plus' in psite_bedgraph_name:
        sign = '+'
    
    if 'minus' in psite_bedgraph_name:
        sign = '-'
    
    print('sign', sign)
                    
    infile = open(psite_bedgraph_name)
    
    for line in infile:
        
        chromo = line.split('\t')[0]
        
        if 'CHR' in chromo.upper():
            chromo = chromo.upper().split('CHR')[1]
        
        if (sign != '+') and (sign != '-'):
            print('sign error: ', sign)
            1/0
        
        if sign == '+':
            nt = int(line.split('\t')[1])
        else:
            nt = int(line.split('\t')[2])
        
        val = float(line.split('\t')[3])
        
        if chromo not in psite_object[sign]:
            psite_object[sign][chromo] = {}
            
        if nt not in psite_object[sign][chromo]:
            psite_object[sign][chromo][nt] = 0
            
        psite_object[sign][chromo][nt] += val
        
    infile.close()
        
    return(psite_object)


"""
Created on Wed May  8 11:11:00 2024

@author: pspea

The premise here is to evaluate the p-site occupancy of CDS associated start codons 
to best determine p-site reading frame resolution on TISs.

---
1. define the CDS start codons
2. identify RPFs that overlap
3. heatmap of reading frame and size

"""

def strand_agnostic(puorf_dict, psite_object):    
    relative_puorf_dict = {}
    start_codons_by_chromo = {}

    for gene in puorf_dict:
        chromo = puorf_dict[gene][-1]['chromo']
                
        if chromo not in start_codons_by_chromo:
            start_codons_by_chromo[chromo] = set()
        
        sign = puorf_dict[gene][-1]['coord_sign']
        
        # 15 nt mORF mask
        if sign == '+':
            mask_stop = puorf_dict[gene][-1]['coord_stop'] - 9
        else:
            mask_stop = puorf_dict[gene][-1]['coord_start'] + 9
        
        #print(gene, chromo, sign)
        
        for puorf in puorf_dict[gene]:
            if puorf == -1:
                puorf_start = puorf_dict[gene][puorf]['coord_start']
                puorf_stop = puorf_dict[gene][puorf]['coord_stop']
            
            if puorf >= 0:
                puorf_start = puorf_dict[gene][puorf]['puorf_start']
                puorf_stop = puorf_dict[gene][puorf]['puorf_stop']
                start_codons_by_chromo[chromo].add(puorf_start)
                            
            rel_start = min(puorf_start, puorf_stop)
            rel_stop = max(puorf_start, puorf_stop)
            puorf_dict[gene][puorf]['left'] = rel_start
            puorf_dict[gene][puorf]['right'] = rel_stop
            puorf_dict[gene][puorf]['mask_stop'] = mask_stop
                                
            if gene not in relative_puorf_dict:
                relative_puorf_dict[gene] = {}
                
            if puorf not in relative_puorf_dict[gene]:
                relative_puorf_dict[gene][puorf] = puorf_dict[gene][puorf]
                
            psite_vals = []
            rnt_coord = []

            length = abs(puorf_stop - puorf_start)
                       
            if sign == '+':
                for rnt in range(-15, length+1):
                    nt = rnt + rel_start
                    
                    if nt in psite_object[sign][chromo]:
                        if nt < mask_stop:
                            psite_vals.append(psite_object[sign][chromo][nt])
                            rnt_coord.append(rnt)
                        else:
                            psite_vals.append(0)
                            rnt_coord.append(rnt)
                        
                    else:
                        psite_vals.append(0)
                        rnt_coord.append(rnt)
                        
            if sign == '-':
                for rnt in range(15, -1*(length+1), -1):
                    nt = rnt + rel_stop
                    if nt in psite_object[sign][chromo]:
                        if nt > mask_stop:
                            psite_vals.append(psite_object[sign][chromo][nt])
                        else:
                            psite_vals.append(0)
                    else:
                        psite_vals.append(0)
                        
                #psite_vals = psite_vals[::-1]
                for rnt in range(-15, (-15+len(psite_vals))):
                    rnt_coord.append(rnt)
                    
            relative_puorf_dict[gene][puorf]['psite_vals'] = psite_vals
            relative_puorf_dict[gene][puorf]['rnt_coord'] = rnt_coord
            
            psite_dict = {}
            
            for i in range(len(rnt_coord)):
                rnt = rnt_coord[i]
                val = psite_vals[i]
                
                psite_dict[rnt] = val
                
            relative_puorf_dict[gene][puorf]['psite_dict'] = psite_dict
                            
    return(relative_puorf_dict)

def if_in_else_zero(num, psite_dict):
    if num in psite_dict:
        return(psite_dict[num])
    else:
        return(0)
    
def detect_overflow(puorf_features, start_codons_by_chromo):
    for uid in puorf_features:
        puorf_features[uid]['lump'] = 'peak'
        
        if puorf_features[uid]['pass_filter_one']:
            #test from left
            psite_dict = puorf_features[uid]['_drop_psite_dict']
            tis_val = if_in_else_zero(0, psite_dict)
            tis_one_up = if_in_else_zero(-1, psite_dict)
            tis_two_up = if_in_else_zero(-2, psite_dict)
            tis_three_up = if_in_else_zero(-3, psite_dict)
            
            lump_sum = 0
            for rnt in range(-3, 1):
                if (rnt in psite_dict):
                    lump_sum += psite_dict[rnt]
            puorf_features[uid]['lump_sum'] = lump_sum
            puorf_features[uid]['tis_val'] = tis_val
            
            #using a 10% leniency
            if (1.1*tis_val) >= (lump_sum/2):
                if ((1.1*tis_val >= tis_three_up and 1.1*tis_one_up >= tis_two_up) or (1.1*tis_two_up >= tis_one_up)):
                    chromo = puorf_features[uid]['_chromo']
                    sign = puorf_features[uid]['_sign']
                    
                    start_set = start_codons_by_chromo[chromo]

                    another_option = -1
                    if sign == '+':
                        lookaround_nt = min(puorf_features[uid]['_left'], puorf_features[uid]['_right'])
                        for coord_nt in range(lookaround_nt - 3, lookaround_nt):
                            if coord_nt in start_set:
                                another_option = coord_nt
                            
                    else:
                        lookaround_nt = max(puorf_features[uid]['_left'], puorf_features[uid]['_right'])
                        for coord_nt in range(lookaround_nt + 1, lookaround_nt + 4):
                            if coord_nt in start_set:
                                another_option = coord_nt
                    
                    puorf_features[uid]['another_option'] = another_option
                    
                    puorf_features[uid]['lump'] = 'lump'
                                
            #test from right            
            psite_dict = puorf_features[uid]['_drop_psite_dict']
            tis_val = if_in_else_zero(0, psite_dict)
            tis_one_up = if_in_else_zero(1, psite_dict)
            tis_two_up = if_in_else_zero(2, psite_dict)
            tis_three_up = if_in_else_zero(3, psite_dict)
            
            lump_sum = 0
            for rnt in range(4):
                if (rnt in psite_dict):
                    lump_sum += psite_dict[rnt]
            puorf_features[uid]['lump_sum'] = lump_sum
        
            if (1.1*tis_val) >= (lump_sum/2):
                if ((1.1*tis_val >= tis_three_up and 1.1*tis_one_up >= tis_two_up) or (1.1*tis_two_up >= tis_one_up)):
                    chromo = puorf_features[uid]['_chromo']
                    sign = puorf_features[uid]['_sign']
                    
                    start_set = start_codons_by_chromo[chromo]

                    another_option = -1
                    if sign == '+':
                        lookaround_nt = min(puorf_features[uid]['_left'], puorf_features[uid]['_right'])
                        for coord_nt in range(lookaround_nt - 3, lookaround_nt):
                            if coord_nt in start_set:
                                another_option = coord_nt
                            
                    else:
                        lookaround_nt = max(puorf_features[uid]['_left'], puorf_features[uid]['_right'])
                        for coord_nt in range(lookaround_nt + 1, lookaround_nt + 4):
                            if coord_nt in start_set:
                                another_option = coord_nt
                    
                    puorf_features[uid]['another_option'] = another_option
                    
                    puorf_features[uid]['lump'] = 'lump'
                            
    return(puorf_features)
        
def build_features(puorf_features, relative_puorf_dict, validated_uorfs):
    
    if len(validated_uorfs) > 0:
        outline = ('Quantifying features ...')
        print(outline)
    else:
        outline = ('Estimating null model features ...')
        print(outline)
    
    for gene in relative_puorf_dict:
        # Not sure if these should be enabled because of the TL bias in May 2023        
        # sign = relative_puorf_dict[gene][-1]['coord_sign']
        # if sign == '+':
        #     tss_nt = relative_puorf_dict[gene][-1]['coord_start']
        # else:
        #     tss_nt = relative_puorf_dict[gene][-1]['coord_stop']
        
        for puorf in relative_puorf_dict[gene]:
            if puorf >= 0:
                uid = len(puorf_features)
                puorf_features[uid] = {}
                relative_puorf_dict[gene][puorf]['uid'] = uid

                puorf_features[uid]['_puorf'] = puorf
                puorf_features[uid]['_gene'] = gene
                puorf_features[uid]['_chromo'] = relative_puorf_dict[gene][-1]['chromo']
                puorf_features[uid]['_left'] = relative_puorf_dict[gene][puorf]['left']
                puorf_features[uid]['_right'] = relative_puorf_dict[gene][puorf]['right']
                puorf_features[uid]['_sign'] = relative_puorf_dict[gene][-1]['coord_sign']
                puorf_features[uid]['_mask_stop'] = relative_puorf_dict[gene][-1]['mask_stop']
                puorf_features[uid]['start_codon']= relative_puorf_dict[gene][puorf]['start_codon']
                                                
                for vuorf in validated_uorfs:
                    '''
                    {'gene': gene,
                     'start': uORFstart,
                     'stop': uORFend,
                     'sign': sign,
                     'is_uorf': signif,
                     'source': 'may_2023',
                     'validated': 'molecular'}
                    '''
                    if gene == validated_uorfs[vuorf]['gene']:
                        is_uorf = validated_uorfs[vuorf]['is_uorf']
                        #print(is_uorf, type(is_uorf))


                        if puorf_features[uid]['_sign'] == '+':

                            if puorf_features[uid]['_left'] == validated_uorfs[vuorf]['start']:
                                puorf_features[uid]['_is_uorf'] = is_uorf
                        else:
                            if puorf_features[uid]['_right'] == validated_uorfs[vuorf]['stop']:
                                puorf_features[uid]['_is_uorf'] = is_uorf

                psite_dict = relative_puorf_dict[gene][puorf]['psite_dict']
                puorf_features[uid]['_drop_psite_dict'] = psite_dict
                                
                # calc total depth
                val_sum = 0
                for rnt in psite_dict:                    
                    if rnt >= 0:
                        val_sum += psite_dict[rnt]
                puorf_features[uid]['puorf_all_frames'] = val_sum
                
                # calc total depth inframe
                val_sum = 0
                for rnt in psite_dict:                    
                    if (rnt >= 0) and rnt % 3 == 0:
                        val_sum += psite_dict[rnt]
                puorf_features[uid]['puorf_inframe'] = val_sum
                
                # calc total depth inframe
                val_sum = 0
                for rnt in psite_dict:                    
                    if (rnt >= 0) and rnt % 3 != 0:
                        val_sum += psite_dict[rnt]
                puorf_features[uid]['puorf_outframe'] = val_sum
                
                # calc initialzation 
                psite_body_all_frames = []
                for rnt in psite_dict:
                    if rnt >= 13:
                        if rnt in psite_dict:
                            psite_body_all_frames.append(psite_dict[rnt])
                
                if len(psite_body_all_frames) > 0:            
                    puorf_features[uid]['puorf_body_all_frames'] = sum(psite_body_all_frames)
                    puorf_body_all_frames = sum(psite_body_all_frames)
                                    
                    puorf_body_median = np.mean([x for x in psite_body_all_frames if x >= 0])
                    puorf_features[uid]['puorf_body_median'] = puorf_body_median
                else:
                    puorf_features[uid]['puorf_body_all_frames'] = 1
                    puorf_body_all_frames = 1
                                    
                    puorf_body_median = 1
                    puorf_features[uid]['puorf_body_median'] = 1                    
                                                                
                # calc initialzation 
                val_sum = 0
                for rnt in range(12+1):
                    if rnt >= 0:
                        if rnt in psite_dict:
                            val_sum += psite_dict[rnt]
                puorf_features[uid]['initialzation_all_frames'] = val_sum
                
                # calc initialzation inframe
                val_sum = 0
                for rnt in range(12+1):
                    if (rnt >= 0) and (rnt % 3 == 0):
                        if rnt in psite_dict:
                            val_sum += psite_dict[rnt]
                puorf_features[uid]['initialzation_inframe'] = val_sum
                
                # calc initialzation outframe
                val_sum = 0
                for rnt in range(12+1):
                    if (rnt >= 0) and (rnt % 3 != 0):
                        if rnt in psite_dict:
                            val_sum += psite_dict[rnt]
                puorf_features[uid]['initialzation_outframe'] = val_sum
                
                # calc TIS
                val_sum = 0
                for rnt in range(3+1):
                    if rnt >= 0:
                        if rnt in psite_dict:
                            val_sum += psite_dict[rnt]
                puorf_features[uid]['tis_all_frames'] = val_sum
                              
                # calc TIS inframe
                val_sum = 0
                for rnt in set([0]):
                    val_sum += psite_dict[rnt]
                puorf_features[uid]['tis_inframe'] = val_sum
                
                # calc TIS out of frame
                val_sum = 0
                for rnt in range(3+1):
                    if (rnt >= 0) and (rnt % 3 != 0):
                        if rnt in psite_dict:
                            val_sum += psite_dict[rnt]
                puorf_features[uid]['tis_outframe'] = val_sum
                
                # calc upstream 
                val_sum = 0
                for rnt in range(-15, 0):
                    if rnt in psite_dict:
                        val_sum += psite_dict[rnt]
                puorf_features[uid]['upstream_all_frames'] = val_sum
                              
                # calc upstream inframe
                val_sum = 0
                for rnt in range(-15, 0):
                    if (rnt in psite_dict) and (rnt % 3 == 0):
                        val_sum += psite_dict[rnt]
                puorf_features[uid]['upstream_inframe'] = val_sum
                
                # calc upstream outframe
                val_sum = 0
                for rnt in range(-15, 0):
                    if (rnt in psite_dict) and (rnt % 3 != 0):
                        val_sum += psite_dict[rnt]
                puorf_features[uid]['upstream_outframe'] = val_sum
                
                # calc heursitic score
                tis_inframe = puorf_features[uid]['tis_inframe']
                tis_outframe = puorf_features[uid]['tis_outframe']
                tis_diff = (tis_inframe - tis_outframe)
                puorf_features[uid]['tis_diff'] = tis_diff
                
                initialzation_inframe = puorf_features[uid]['initialzation_inframe']
                initialzation_outframe = puorf_features[uid]['initialzation_outframe']
                initialzation_all_frames = puorf_features[uid]['initialzation_all_frames']
                init_diff = (initialzation_inframe - initialzation_outframe)
                puorf_features[uid]['init_diff'] = init_diff

                upstream_inframe = puorf_features[uid]['upstream_inframe']   
                upstream_outframe = puorf_features[uid]['upstream_outframe']
                upstream_diff = (upstream_inframe - upstream_outframe)
                puorf_features[uid]['upstream_diff'] = upstream_diff
                upstream_all_frames = puorf_features[uid]['upstream_all_frames']
                
                puorf_inframe = puorf_features[uid]['puorf_inframe']
                puorf_outframe = puorf_features[uid]['puorf_outframe']
                #puorf_all_frames = puorf_features[uid]['puorf_all_frames']
                puourf_diff = (puorf_inframe - puorf_outframe)
                puorf_features[uid]['puourf_diff'] = puourf_diff

                nt_length = abs(relative_puorf_dict[gene][puorf]['right'] - 
                       relative_puorf_dict[gene][puorf]['left'])
                puorf_features[uid]['aa_length'] = nt_length/3
                
                #continuity
                continuity_list = []
                
                for i in range(max(psite_dict)):
                    codon_start = 0+i*3
                    codon_stop = 3+i*3
                    temp_bins = {0:0, 1:0, 2:0}
                                        
                    if codon_stop in psite_dict:
                        for rnt in range(codon_start, codon_stop):
                            inbin = rnt % 3
                            val = psite_dict[rnt]
                            temp_bins[inbin] += val
                            #print(rnt, inbin, val)
                                                    
                        if (temp_bins[0] > temp_bins[1]) and (temp_bins[0] > temp_bins[2]):
                            continuity_list.append(1)
                            #print(1)
                        else:
                            continuity_list.append(0)
                            #print(0)
                                                            
                if len(continuity_list) > 0:
                    continuity = sum(continuity_list)-len(continuity_list)
                    consistency = sum(continuity_list)/len(continuity_list)
                else:
                    continuity = 0
                    consistency = 0
                
                puorf_features[uid]['continuity'] = continuity
                puorf_features[uid]['consistency'] = consistency
                
                #overflow
                # calc upstream 
                initialzation_overflow = 0
                for rnt in range(-15, 15+1):
                    if rnt in psite_dict:
                        initialzation_overflow += psite_dict[rnt]
                puorf_features[uid]['initialzation_overflow'] = initialzation_overflow
                
                puorf_all_overflow = 0
                for rnt in psite_dict:
                    puorf_all_overflow += psite_dict[rnt]
                puorf_features[uid]['puorf_all_overflow'] = puorf_all_overflow
                
                if puorf_all_overflow > 0:
                    overflow_weight =  initialzation_inframe * (initialzation_overflow/puorf_all_overflow)
                else:
                    overflow_weight = 0
                    
                hscore = (tis_diff-puorf_body_median) + (init_diff-puorf_body_median) + (puourf_diff*consistency) + (puorf_inframe - puorf_body_all_frames) - (upstream_diff + upstream_all_frames)
                puorf_features[uid]['hscore'] = hscore
                
                lscore = (tis_inframe - puorf_body_median) + (puorf_inframe - puorf_body_median) + overflow_weight + (initialzation_all_frames - puorf_body_all_frames)
                puorf_features[uid]['lscore'] = lscore
                '''
                #
                '''
    
    for uid in puorf_features:
        puorf_features[uid]['pass_filter_one'] = True
    
    start_codons_by_chromo = {}
    for uid in puorf_features:
        if puorf_features[uid]['pass_filter_one']:
            chromo = puorf_features[uid]['_chromo']
            sign = puorf_features[uid]['_sign']
            
            if sign == '+':
                start_nt = puorf_features[uid]['_left']
            else:
                start_nt = puorf_features[uid]['_right']
                
            if chromo not in start_codons_by_chromo:
                start_codons_by_chromo[chromo] = set()
            
            start_codons_by_chromo[chromo].add(start_nt)
                    
    puorf_features = detect_overflow(puorf_features, start_codons_by_chromo)
        
    for uid in puorf_features:
        
        # if puorf_features[uid]['lscore'] > puorf_features[uid]['hscore']:
        #     puorf_features[uid]['resolved_score'] = puorf_features[uid]['lscore']
        # else:
        #     puorf_features[uid]['resolved_score'] = puorf_features[uid]['hscore']
        
        if puorf_features[uid]['lump'] == 'lump':
            reso = puorf_features[uid]['lscore']
        else:
            reso = puorf_features[uid]['hscore']
            
        puorf_features[uid]['resolved_score'] = reso
        
        '''checking
        gene = puorf_features[uid]['_gene']
        start = puorf_features[uid]['_right']
        if (gene == 'YEL009C') and start == 140124: 
        #if (gene == 'YKR042W'): 
            print(gene)
            print(puorf_features[uid])
            #print('#')
            #print(relative_puorf_dict[gene])
            1/0
        '''
                
    return(relative_puorf_dict, puorf_features)

def features_to_bed(features_dict, start_codon_counter, cut_off):
    output = resources_dict['output']
    
    cut_off_lookup_dict = {}
    cut_off_lookup = cut_off * 2
    
    for is_lump in start_codon_counter:
        if is_lump not in cut_off_lookup_dict:
            cut_off_lookup_dict[is_lump] = {}
            
        for start_codon in start_codon_counter[is_lump]:
            if start_codon not in cut_off_lookup_dict[is_lump]:
                cut_off_lookup_dict[is_lump][start_codon] = 0 
                
            fdr_list = start_codon_counter[is_lump][start_codon]
            
            fdr_list.sort(reverse = True)
            
            cut_off_val = fdr_list[int(cut_off_lookup*len(fdr_list))]
            
            cut_off_lookup_dict[is_lump][start_codon] = cut_off_val
                        
    for uid in features_dict:
        features_dict[uid]['fdr'] = 1
        
        if features_dict[uid]['pass_filter_one']:
            is_lump = features_dict[uid]['lump']            
            start_codon = features_dict[uid]['start_codon']
            
            rs = features_dict[uid]['resolved_score']
            
            cut_off_val = cut_off_lookup_dict[is_lump][start_codon]
            
            if rs > cut_off_val:
            
                fdr_list = start_codon_counter[is_lump][start_codon]
                
                numer = len([fdrs for fdrs in fdr_list if fdrs > rs])
                fdr_val = numer/len(fdr_list)
                
                features_dict[uid]['fdr'] = fdr_val
                
                #print(uid, rs, cut_off_val, fdr_val)
                
            else:
                features_dict[uid]['fdr'] = cut_off_lookup

    cm = {'Tt':0, 'Tf': 0, 'Ft': 0, 'Ff': 0}
    for uid in features_dict:
        features_dict[uid]['cm']='xx'
        
        if '_is_uorf' in features_dict[uid]:
            #if features_dict[uid]['pass_filter_one']:
                if features_dict[uid]['fdr'] <= cut_off:
                    if features_dict[uid]['_is_uorf']:
                        cm['Tt'] += 1 
                        features_dict[uid]['cm']='Tt'
                    else:
                        cm['Tf'] += 1
                        features_dict[uid]['cm']='Tf'
                else:
                    if features_dict[uid]['_is_uorf']:
                        cm['Ft'] += 1 
                        features_dict[uid]['cm']='Ft'
                    else:
                        cm['Ff'] += 1
                        features_dict[uid]['cm']='Ff'

    print(cm)
    
    temp_bed_name = ('{output}_temp_{cut_off}.bed').format(output = output, cut_off = cut_off)
    temp_bed = open(temp_bed_name, 'w')
    
    for uid in features_dict:
        fdr = features_dict[uid]['fdr']
        pfo = features_dict[uid]['pass_filter_one']
        
        if (features_dict[uid]['cm'] != 'xx'):            
            outline = ('{chromo}\t{left}\t{right}\t{name}.{left}.{right}.{lump}.{pfo}.{cm}\t{fdr}\n').format(
                chromo = features_dict[uid]['_chromo'],
                left = features_dict[uid]['_left']-1,
                right = features_dict[uid]['_right'],
                name = features_dict[uid]['_gene'],
                lump = puorf_features[uid]['lump'],
                pfo = pfo,
                cm = features_dict[uid]['cm'],
                fdr = fdr)
            temp_bed.write(outline)
            
    temp_bed.close()
    
    return(cut_off_lookup_dict)

def predictions_to_bed(features_dict, cut_off):
    run_name = args.run_name
    
    temp_bed_name = ('{run_name}/{run_name}_predictions_{cut_off}.bed').format(
        run_name = run_name, cut_off = cut_off)
    temp_bed = open(temp_bed_name, 'w')
    
    outline = ('Model predictions exported as bed file : {}').format(temp_bed_name)
    
    for uid in features_dict:
        predict_score = features_dict[uid]['predict']
        
        #if (fdr <= cut_off) or (features_dict[uid]['cm'] != 'xx'):
        if predict_score >= cut_off:            
            outline = ('{chromo}\t{left}\t{right}\t{name}.{left}.{right}.{cm}\t{predict_score}\t{sign}\n').format(
                chromo = features_dict[uid]['_chromo'],
                left = features_dict[uid]['_left']-1,
                right = features_dict[uid]['_right'],
                name = features_dict[uid]['_gene'],
                cm = features_dict[uid]['cm'],
                predict_score = predict_score,
                sign = features_dict[uid]['_sign'])
            temp_bed.write(outline)
            
    temp_bed.close()
    
    temp_bed_name = ('{run_name}/{run_name}_TIS_predictions_{cut_off}.bed').format(
        run_name = run_name, cut_off = cut_off)
    temp_bed = open(temp_bed_name, 'w')
    
    outline = ('Model predictions exported as bed file : {}').format(temp_bed_name)
    
    for uid in features_dict:
        predict_score = features_dict[uid]['predict']
        
        sign = features_dict[uid]['_sign']
        if sign == '+':
            left = features_dict[uid]['_left']-1
            right = left + 3
        else:
            right = features_dict[uid]['_right']
            left = right - 2
        
        #if (fdr <= cut_off) or (features_dict[uid]['cm'] != 'xx'):
        if predict_score >= cut_off:            
            outline = ('{chromo}\t{left}\t{right}\t{name}.{left}.{right}.{cm}\t{predict_score}\t{sign}\n').format(
                chromo = features_dict[uid]['_chromo'],
                left = left,
                right = right,
                name = features_dict[uid]['_gene'],
                cm = features_dict[uid]['cm'],
                predict_score = predict_score,
                sign = sign)
            temp_bed.write(outline)
            
    temp_bed.close()
            
# Define model
# for torch
class zagreus(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(n_inputs, 3*n_inputs)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(3*n_inputs, 4*n_inputs)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(4*n_inputs, 3*n_inputs)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(3*n_inputs, 2*n_inputs)
        self.act4 = nn.ReLU()
        self.layer5 = nn.Linear(2*n_inputs, n_inputs)
        self.act5 = nn.ReLU()
        self.output = nn.Linear(n_inputs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act4(self.layer4(x))
        x = self.act5(self.layer5(x))
        x = self.sigmoid(self.output(x))
        return x

'''
#
'''

#step 1 - make resource object
resources_dict = parse_resource_object()

if args.build_genome_object:
    print('Building genome object ...')
    
    fa_dict, coord_dict = make_TL_fasta()
    puorf_dict = demarcate_regions(fa_dict, coord_dict)
    
    genome_object = resources_dict['genome_object']
    puorf_file = open(genome_object, 'wb')
    pickle.dump(puorf_dict, puorf_file)
    
    outline = ('Completed genome object: {}').format(genome_object)
    print(outline)
    
if args.parse_read_object:
    sam_filename = resources_dict['sam_filename']
    run_name = args.run_name
    
    outline = ('Parsing RPF sam file(s): {}').format(sam_filename)
    print(outline)
    
    bedgraph_set = set()
    
    # TODO make data type CLI
    # TODO this is being overwrote latter
    sample_read_object = {}
    
    #integrate sam_filename(s) into experimental object
    if ',' in sam_filename:
        sam_filename_list = sam_filename.split(',')
        for each_sam_filename in sam_filename_list:
            each_sam_filename = each_sam_filename.strip()
            sample_read_object, bedgraph_set = parse_reads(sample_read_object, each_sam_filename, bedgraph_set)
            
    else:
        sample_read_object, bedgraph_set = parse_reads(sample_read_object, sam_filename, bedgraph_set)
        
    print(resources_dict)
    
    print('bedgraph_set', bedgraph_set)
    
    bedgraph_filenames = resources_dict['bedgraph_filenames']
    bedgraph_file = open(bedgraph_filenames, 'wb')
    pickle.dump(bedgraph_set, bedgraph_file)
    
    # read_object_filename = resources_dict['read_object']
    # read_object_file = open(read_object_filename, 'wb')
    # pickle.dump(sample_read_object, read_object_file)
    
    #resources_dict['bedgraph_filenames'].append(psite_bedgraph_name)
    # bedgraph_filenames_file = open(bedgraph_filenames, 'wb')
    # pickle.dump(bedgraph_dict, bedgraph_filenames_file)
        
    # outline = ('Completed read object: {}').format(read_object_filename)
    # print(outline)
    
    # print('The following bedgraphs were generated:\n')
    # for sign in bedgraph_dict:
    #     for size in bedgraph_dict[sign]:
    #         each = bedgraph_dict[sign][size]
    #         outline = ('\t{each}\n').format(each = each)
    #         print(outline)
    
if args.build_features_object:
    outline = ("Beginning to build and score features...")
    print(outline)
    
    psite_bedgraph_names = resources_dict['bedgraph_filenames']
    highqc_candidate_cutoff = float(resources_dict['highqc_candidate_cutoff'])
    
    bedgraph_filenames = resources_dict['bedgraph_filenames']
    bedgraph_file = open(bedgraph_filenames, 'rb')
    bedgraph_set = pickle.load(bedgraph_file)
    
    genome_object = resources_dict['genome_object']
    puorf_file = open(genome_object, 'rb')
    puorf_dict = pickle.load(puorf_file)
    
    parse_uorf_object = resources_dict['parse_uorf_object']
    
    if parse_uorf_object:
        #validated_uorfs = parse_uorf()
        validated_uorfs = parse_gff_uorf()
    else:
        print('... skipping validated uORFs')
        validated_uorfs = set()
        
    psite_object = {'+':{},
                    '-':{}}
    for psite_bedgraph_name in bedgraph_set:
        psite_object = load_psite_object(psite_bedgraph_name, psite_object)
    
    relative_puorf_dict = strand_agnostic(puorf_dict, psite_object)
    
    puorf_features = {}
    relative_puorf_dict, puorf_features = build_features(puorf_features, relative_puorf_dict, validated_uorfs)
    
    fa_dict, coord_dict = make_TL_fasta()
    start_codon_counter = calculate_scramble_pval(fa_dict, coord_dict)
    
    # using the scramble as a fdr calc and set the qc
    cut_off_lookup_dict = features_to_bed(puorf_features, start_codon_counter, highqc_candidate_cutoff)

    features_object_filename = resources_dict['features_object_filename']
    df = pd.DataFrame.from_dict(puorf_features, orient='index')
    df = df.drop(columns = ['_drop_psite_dict'])
    df.to_csv(features_object_filename, index=False)
    
    #puorf_features
    '''
    # here we're going to clean up the feature df to strip out the columns that would ruin the ML training loop
    '''
    train_object_features_filename = resources_dict['train_object_features']
    train_test_df = df.loc[(df['cm'] == 'Tt') | (df['cm'] == 'Ff') | (df['cm'] == 'Sf')]
    train_test_df = train_test_df.replace({'Ff': 0, 'Sf':0, 'Tt': 1})
    train_test_df = train_test_df.drop(columns = ['_puorf', '_gene', '_is_uorf', '_chromo',
                                       '_left', '_right', '_sign', '_mask_stop', 
                                       'start_codon', 'hscore', 'lscore', 'pass_filter_one',
                                       'lump', 'another_option', 'resolved_score', 'fdr'])
    train_test_df.to_csv(train_object_features_filename, header = False)
    #
    predict_object_features_filename = resources_dict['predict_object_features']
    predict_df = df.drop(columns = ['_puorf', '_gene', '_is_uorf', '_chromo',
                                       '_left', '_right', '_sign', '_mask_stop',
                                       'start_codon', 'hscore', 'lscore', 'pass_filter_one',
                                       'lump', 'another_option', 'resolved_score', 'fdr', 'cm'])
    predict_df.to_csv(predict_object_features_filename, header = False)
    
    outline = ("Features built and scored:\n"
               "\tFeatures with fdr scores: {features_object_filename}\n"
               "\tCandidate uORF object for model training: {train_object_features_filename}\n"
               "\tCandidate uORF object for prediction: {predict_object_features_filename}").format(
                   features_object_filename = features_object_filename,
                   train_object_features_filename = train_object_features_filename,
                   predict_object_features_filename = predict_object_features_filename)
    print(outline)

# for torch
if args.train_model:
    train_object_features_filename = resources_dict['train_object_features']
    
    print(train_object_features_filename)
    
    dnn_model_filename = resources_dict['dnn_model_filename']
    
    outline = ('Training model on {}').format(train_object_features_filename)
    print(outline)
        
    # Read data
    data = pd.read_csv(train_object_features_filename, header=0)
    print(data.head())
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    print(X.head())
    # Binary encoding of labels
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # y = encoder.transform(y)
    
    # Convert to 2D PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    n_inputs = X.size()[1]
        
    # Helper function to train one model
    def model_train(model, ml_X_train, ml_y_train, ml_X_val, ml_y_val):
        # loss function and optimizer
        loss_fn = nn.BCELoss()  # binary cross entropy
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
        n_epochs = 300   # number of epochs to run
        batch_size = 100  # size of each batch
        batch_start = torch.arange(0, len(ml_X_train), batch_size)
    
        # Hold the best model
        best_acc = - np.inf   # init to negative infinity
        best_weights = None
    
        for epoch in range(n_epochs):
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                
                for start in bar:
                    # take a batch
                    X_batch = ml_X_train[start:start+batch_size]
                    y_batch = ml_y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    acc = (y_pred.round() == y_batch).float().mean()
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(ml_X_val)
            acc = (y_pred.round() == ml_y_val).float().mean()
            acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        return best_acc
    
    # train-test split: Hold out the test set for final model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
    
    # define 5-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cv_scores_deep = []
    
    for train, test in kfold.split(X_train, y_train):
        # create model, train, and get accuracy
        model = zagreus()
        acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
        print("Accuracy (deep): %.2f" % acc)
        cv_scores_deep.append(acc)
    
    deep_acc = np.mean(cv_scores_deep)
    deep_std = np.std(cv_scores_deep)
    
    print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))
    
    # rebuild model with full set of training data
    model = zagreus()
    acc = model_train(model, X_train, y_train, X_test, y_test)
    print(f"Final model accuracy: {acc*100:.2f}%")
    
    model.eval()
    
    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(5):
            y_pred = model(X_test[i:i+1])
            print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")
            
    torch.save(model.state_dict(), dnn_model_filename)
    
    outline = ('Model saved as: {}').format(dnn_model_filename)
    print(outline)
    
# for torch
if args.predict:
            
    # Read data
    predict_object_features_filename = resources_dict['predict_object_features']
    features_object_filename = resources_dict['features_object_filename']
    cutoff = float(resources_dict['prediction_cutoff'])
    run_name = args.run_name
    
    outline = ('Model performing predictions on {}').format(predict_object_features_filename)
    print(outline)
        
    data = pd.read_csv(predict_object_features_filename, header=None)
    index = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    predict = torch.tensor(X.values, dtype=torch.float32)
    
    n_inputs = predict.size()[1]
    
    dnn_model_filename = resources_dict['dnn_model_filename']
    model = zagreus()
    model.load_state_dict(torch.load(dnn_model_filename))
    model.eval()
    
    results = model(predict)
    results.size()
    isitarray=results.detach().numpy()
    
    features_df = pd.read_csv(features_object_filename)
    features_df['predict'] = isitarray
    
    results_file_name = ('{run_name}/{run_name}_predictions.csv').format(run_name = run_name)
    features_df.to_csv(results_file_name, index=False)
    
    outline = ('Model predictions saved as: {}').format(results_file_name)
    print(outline)
    
    predictions = features_df.to_dict('index')
        
    predictions_to_bed(predictions, cutoff)
    
if args.run_analysis:
    run_name = args.run_name
    
    ''' Handle config setup '''
    bashCommand = ('python uorfish_v0.9.py -name {run_name} -config').format(
                       run_name = run_name)
    #print(bashCommand)
    
    if args.sam_filename:
        bashCommand += (' -sam {}').format(args.sam_filename)
        print(bashCommand)
        
    if args.transcript_leader_filename:
        bashCommand += (' -tl {}').format(args.transcript_leader_filename)
        print(bashCommand)
                
    if args.fasta_reference_file:
        bashCommand += (' -fa {}').format(args.fasta_reference_file)
        print(bashCommand)
    
    if args.dubious_orfs_filename:
        bashCommand += (' -dubious {}').format(args.dubious_orfs_filename)
        print(bashCommand)
        
    if args.genome_object:
        bashCommand += (' -go {}').format(args.genome_object)
        print(bashCommand)
        
    if args.genome_gff_object:
        bashCommand += (' -go_gff {}').format(args.genome_gff_object)
        print(bashCommand)
        
    if args.read_object:        
        bashCommand += (' -ro {}').format(args.read_object)
        print(bashCommand)
                
    if args.uorf_validation_type:
        bashCommand += (' -uvt {}').format(args.uorf_validation_type)
        print(bashCommand)
        
    if args.validated_uorfs_filename:
        bashCommand += (' -val_uorfs {}').format(args.validated_uorfs_filename)
        print(bashCommand)
    
    if args.features_object_filename:
        bashCommand += (' -fo {}').format(args.features_object_filename)
        print(bashCommand)
        
    if args.train_object_features:
        bashCommand += (' -to {}').format(args.train_object_features)
        print(bashCommand)
        
    if args.predict_object_features:
        bashCommand += (' -po {}').format(args.predict_object_features)
        print(bashCommand)
             
    if args.dnn_model_filename:
        bashCommand += (' -model {}').format(args.dnn_model_filename)
        print(bashCommand)
                
    if args.prediction_cutoff:
        bashCommand += (' -cutoff {}').format(args.prediction_cutoff)
        print(bashCommand)
        
    subprocess.run([bashCommand],stderr=subprocess.STDOUT,shell=True)
    
    ''' Handle Genome object '''
    bashCommand = ('python uorfish_v0.9.py -name {run_name} -genome').format(
                       run_name = run_name)
    print(bashCommand)
    subprocess.run([bashCommand],stderr=subprocess.STDOUT,shell=True)
    
    ''' Handle RPF object '''
    bashCommand = ('python uorfish_v0.9.py -name {run_name} -rpf').format(
                       run_name = run_name)
    print(bashCommand)
            
    if args.parse_read_object:
        bashCommand += (' -read')
        print(bashCommand)
        
    if args.make_bedgraph:
        bashCommand += (' -bedgraph')
        print(bashCommand)
        
    subprocess.run([bashCommand],stderr=subprocess.STDOUT,shell=True)
    
    ''' Handle features '''    
    bashCommand = ('python uorfish_v0.9.py -name {run_name} -features').format(
                       run_name = run_name)
    print(bashCommand)
    
    if args.parse_uorf_object:
        bashCommand += (' -known')
        print(bashCommand)
        
    subprocess.run([bashCommand],stderr=subprocess.STDOUT,shell=True)
        
    ''' Handle training '''
    bashCommand = ('python uorfish_v0.9.py -name {run_name} -train').format(
                       run_name = run_name)
    print(bashCommand)
    subprocess.run([bashCommand],stderr=subprocess.STDOUT,shell=True)
    
    ''' Handle prediction '''
    bashCommand = ('python uorfish_v0.9.py -name {run_name} -predict').format(
                       run_name = run_name)
    print(bashCommand)
    subprocess.run([bashCommand],stderr=subprocess.STDOUT,shell=True)
        


# predictions_filename = ('C:/Gresham/tiny_projects/uorfish/test/test_predictions.csv')

# df = pd.read_csv(predictions_filename)
    
# predictions = df.to_dict('index')

# #maximize_cm = {'Tt':0, 'Tf': 0, 'Ft': 0, 'Ff': 0}
# max_score = -1*np.inf
# score_list = []
# opt_threshold = 0

# for i in range(0,101):
#     threshold = i/100
#     maximize_cm = {'Tt':0, 'Tf': 0, 'Ft': 0, 'Ff': 0}
    
#     for puorf in predictions:
#         if predictions[puorf]['cm'] != 'xx':
#             if predictions[puorf]['predict'] >= threshold:
#                 if predictions[puorf]['_is_uorf']:
#                     maximize_cm['Tt'] += 1
#                 else:
#                     maximize_cm['Tf'] += 1
#             else:
#                 if predictions[puorf]['_is_uorf']:
#                     maximize_cm['Ft'] += 1
#                 else:
#                     maximize_cm['Ff'] += 1
#     tpr = maximize_cm['Tt']/(maximize_cm['Tt'] + maximize_cm['Ft'])
#     fpr = maximize_cm['Tf']/(maximize_cm['Ff'] + maximize_cm['Tf'])
#     score =  tpr / fpr
#     score_list.append(score)
    
#     #score = -1* (maximize_cm['Ft'] + maximize_cm['Tf'])
    
#     if score >= max_score:
#         print(threshold, score, maximize_cm)
#         max_score = score
#         opt_threshold = threshold
        
# from numpy import trapz


# # The y values.  A numpy array is used here,
# # but a python list could also be used.
# y = np.array(score_list)

# # Compute the area using the composite trapezoidal rule.
# area = trapz(y, dx=1)
# print("area =", area)
