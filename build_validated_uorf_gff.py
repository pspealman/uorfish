# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:42:22 2024

@author: pspea
"""
search_window = 10000
start_codons = set(['ATG', 'TTG', 'GTG', 'CTG', 'ATA', 'ATT', 'ACG', 'ATC', 'AAG', 'AGG']) # Brar_2020
#start_codons = set(['ATG', 'TTG', 'GTG', 'CTG', 'ATA', 'ATT', 'ACG', 'ATC'])
stop_codons = set(['TAA', 'TAG', 'TGA'])

check_name_to_sign = {'W':'+', 'C':'-'}

alphabet_to_chromo = {'A':'I', 'B':'II', 'C':'III', 'D':'IV', 'E':'V',
                      'F':'VI', 'G':'VII', 'H':'VIII', 'I':'IX', 'J':'X',
                      'K':'XI', 'L':'XII', 'M':'XIII', 'N':'XIV', 'O':'XV',
                      'P':'XVI'}

'''
# load reference genome, use it to scan for stops and starts
'''
fa_dict = {}
fa_file = open('C:/Gresham/genomes/Ensembl/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa')

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

def reverse_complement(sequence):
    sequence = sequence.upper()
    complement = sequence.translate(str.maketrans('ATCG', 'TAGC'))
    return(complement[::-1])

def pull_for_start(chromo, start, sign, exp_start_codon):
    start = int(start)
    match = True
    
    if sign == '+':
        query_seq = fa_dict[chromo][(start-1):(start+2)]
        
    if sign == '-':
        query_seq = fa_dict[chromo][(start-3):(start)]
        query_seq = reverse_complement(query_seq)
        
    if (exp_start_codon != 'NNN') and (exp_start_codon != query_seq):
        match = False
            
    return(query_seq, match)

def pull_for_stop(chromo, stop, sign, exp_stop_codon):
    stop = int(stop)
    match = True
    
    if sign == '+':
        query_seq = fa_dict[chromo][(stop-4):(stop-1)]
        
    if sign == '-':
        query_seq = fa_dict[chromo][(stop-1):(stop+2)]
        query_seq = reverse_complement(query_seq)
        
    if (exp_stop_codon != 'NNN') and (exp_stop_codon != query_seq):
        match = False
            
    return(query_seq, match)

def scan_for_stop(chromo, start, sign):
    start = int(start)
    
    if sign == '+':
        window_seq = fa_dict[chromo][(start-1):(start-1 + (3*search_window))]
        
        for i in range(search_window + 1):
            query_seq = window_seq[(i*3):((i*3)+3)]
            
            if query_seq in stop_codons:
                stop_nt = ( (start-1)+((i*3)+3) )+1
                return(query_seq, stop_nt)
            
        else:
            stop_nt = ( (start-1)+((i*3)+3) )+1
            return('NNN', stop_nt)
        
    if sign == '-':
        leftside = (start-1 - (3*search_window))
        backstop = max(0, leftside)
        window_seq = fa_dict[chromo][(backstop):(start)]
        window_seq = reverse_complement(window_seq)
        
        for i in range(search_window + 1):
            query_seq = window_seq[(i*3):((i*3)+3)]
            if query_seq in stop_codons:
                stop_nt = (start-((i*3)+2))-1
                return(query_seq, stop_nt)
            
        else:
            stop_nt = (start-((i*3)+2))-1
            return('NNN', stop_nt)
            
def output_to_gff(output_filename, left_mod, right_mod, deets, final_output=False):
    
    if not final_output:
        output_file = open(output_filename, 'a')
        
        (chromo, istype, left, right, sign, 
            name, is_uorf, gene, source,
            start_codon, exp_start_codon, start_match,
            stop_codon, exp_stop_codon, stop_match,
            validated) = deets
        
        outline = ('{chromo}\tcandidate_uorf\t{istype}\t{left}\t{right}\t.\t{sign}\t.\t'
                'ID={name}_{source};Parent={gene};Source={source};'
                'aTIS={start_codon};aTIS_exp={exp_start_codon};aTIS_match={start_match};'
                'Stop={stop_codon};Stop_exp={exp_stop_codon};Stop_match={stop_match};'
                'validated={validated};is_uORF={is_uorf}\n').format(
                    chromo = chromo, istype = istype, left = left + left_mod, right = right + right_mod, sign = sign, 
                    name = name, is_uorf = is_uorf, gene = gene, source = source,
                    start_codon = start_codon, exp_start_codon = exp_start_codon, start_match = start_match,
                    stop_codon = stop_codon, exp_stop_codon = exp_stop_codon, stop_match = stop_match,
                    validated = validated)
        
        output_file.write(outline)    
        
    if final_output:
        output_file = open(output_filename, 'a')
        
        (chromo, istype, left, right, sign, 
            name, is_uorf, gene, source,
            start_codon, exp_start_codon, start_match,
            stop_codon, exp_stop_codon, stop_match,
            validated) = deets
        
        outline = ('{chromo}\tvalidated_orf\t{istype}\t{left}\t{right}\t.\t{sign}\t.\t'
                'ID={name}_{source};Parent={gene};Source={source};'
                'aTIS={start_codon};Stop={stop_codon};validated={validated};is_uORF={is_uorf}\n').format(
                    chromo = chromo, istype = istype, left = left + left_mod, right = right + right_mod, sign = sign, 
                    name = name, is_uorf = is_uorf, gene = gene, source = source,
                    start_codon = start_codon, stop_codon = stop_codon, validated = validated)
        
        output_file.write(outline)
        
'''
subprocess to handle may_2023
'''
may_2023_dict = {}

may_2023_file = open('C:/Gresham/tiny_projects/quorf/elife-69611-supp1-v2_pgA_curated.txt')

for line in may_2023_file:
    if line[0] != 'u':
        if line.split('\t')[3] == 'Scer':
            uorf = line.split('\t')[0]
            istype = line.split('\t')[2]
                            
            start_codon = line.split('\t')[12].replace('U', 'T')
            stop_codon = line.split('\t')[16].replace('U', 'T')
            
            if stop_codon == 'NA':
                stop_codon = 'NNN'
            
            may_2023_dict[uorf] = {'start_codon':start_codon,
                                   'stop_codon':stop_codon,
                                   'istype': istype}
may_2023_file.close()
            
def return_may_2023(chromo, left, right, sign):
    global may_2023_dict
    
    if left == 'oORF' or right == 'oORF':
        if left == 'oORF':
            outside_name = ('chr{chromo}:{right}-oORF').format(
                chromo = chromo, right = right)
            #left = int(right) - 2
            
        if right == 'oORF':
            outside_name = ('chr{chromo}:{left}-oORF').format(
                chromo = chromo, left = left)
            
    else:
        if sign == '+':
            outside_name = ('chr{chromo}:{left}-{right}').format(
                chromo = chromo, left = left, right = right)
        else:
            outside_name = ('chr{chromo}:{right}-{left}').format(
                chromo = chromo, left = left, right = right)
    
    start_codon = may_2023_dict[outside_name]['start_codon']
    stop_codon = may_2023_dict[outside_name]['stop_codon']
    istype = may_2023_dict[outside_name]['istype']
    
    return(start_codon, stop_codon, istype, outside_name)

def parse_qc_may_2023(line):
    global may_2023_dict
    
    #print(line)

    _ct, gene, left, right, sign, is_uorf, source, validated, = line.split(',')
            
    chr_alphabet = gene[1]
    chromo = alphabet_to_chromo[chr_alphabet]
            
    exp_start_codon = 'NNN'
    exp_stop_codon = 'NNN'
    istype = '_uORF'
    
    exp_start_codon, exp_stop_codon, istype, outside_name = return_may_2023(chromo, left, right, sign)
    
    if right == 'oORF':
        stop_codon, right = scan_for_stop(chromo, left, sign)
        
    if left == 'oORF':
        stop_codon, left = scan_for_stop(chromo, right, sign)
                    
    left = int(left)
    right = int(right)
    
    if sign == '+':
        start = left
        stop = right
        
    if sign == '-':
        start = right
        stop = left + 1
            
    name = ('{gene}_{start}').format(gene = gene, start = start)
                
    start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
    stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
            
    deets = (chromo, istype, left, right, sign, 
        name, is_uorf, gene, source,
        start_codon, exp_start_codon, start_match,
        stop_codon, exp_stop_codon, stop_match,
        validated)
    
    if start_match == True and stop_match == True:
        if start_codon not in start_codons:
            print('may_2023_uncorrected start_codon')
            print(line)
            print(right, left, start, start_codon)
            1/0
        
        if stop_codon not in stop_codons:
            print('may_2023 _uncorrected stop_codon')
            print(line)
            print(right, left, stop, stop_codon)
            1/0
        
        if sign == '+':
            left_mod = 0
            right_mod = -1
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
            
        output_to_gff(uncorrected_outfile_name, left_mod, right_mod, deets)
        output_to_gff(validated_outfile_name, left_mod, right_mod, deets, True)
    
    if start_match == False or stop_match == False:            
        # if start_codon not in start_codons:
            # print('may_2023 no_match start_codon')
            # print(line)
            # print(right, left, start, start_codon)
            #1/0
            
        # if stop_codon not in stop_codons:
            # print('may_2023 no_match stop_codon')
            # print(line)
            # print(right, left, stop, stop_codon)
            #1/0
            
        #output precorrection
        output_to_gff(mismatch_outfile_name, 0, 0, deets)
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
                
        left = left + left_mod
        right = right + right_mod
        
        if sign == '+':
            stop_codon, right = scan_for_stop(chromo, left, sign)
        
        if sign == '-':
            stop_codon, left = scan_for_stop(chromo, right, sign)
            
        if sign == '+':
            start = left
            stop = right
            
        if sign == '-':
            start = right
            stop = left + 1
                
        start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
        stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
        
        if (start_codon not in start_codons) or (not start_match):
            print('may_2023 corrected start_codon')
            print(line)
            print(right, left, start, start_codon, exp_start_codon)
            1/0
            
        if (stop_codon not in stop_codons) or (not stop_match):
            print('may_2023 corrected stop_codon')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon) 
            1/0
        
        deets = (chromo, istype, left, right, sign, 
            name, is_uorf, gene, source,
            start_codon, exp_start_codon, start_match,
            stop_codon, exp_stop_codon, stop_match,
            validated)
        
        if sign == '+':
            left_mod = 0
            right_mod = -1
        
        if sign == '-':
            left_mod = 0
            right_mod = -1
            print('There should be none of these - 11.07.2024')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon)
            1/0
        
        output_to_gff(corrected_outfile_name, left_mod, right_mod, deets)
        output_to_gff(validated_outfile_name, left_mod, right_mod, deets, True)
        
    #return(may_2023_mismatch)

'''
subprocess to handle Eisenberg_2020

'''
def parse_qc_eisenberg_2020():
    gff_file = open("C:/Gresham/genomes/Ensembl/Saccharomyces_cerevisiae.R64-1-1.50.gff3")
    
    coordinates_dict = {}
    
    for line in gff_file:
        if line[0] != "#":        
            if 'CDS' in line.split('\t')[2]:
                #I	sgd	CDS	335	649	.	+	0	ID=CDS:YAL069W;Parent=transcript:YAL069W_mRNA;protein_id=YAL069W
                chromo, _source, region, left, right, _dot, sign, _score, deets = line.split('\t')
                left = int(left)
                right = int(right)
                
                gene = deets.split('ID=CDS:')[1].split(';')[0]
                
                if gene not in coordinates_dict:
                    coordinates_dict[gene] = {'chromo': chromo,
                                              'left': left,
                                              'right': right,
                                              'sign': sign}
                    
                if left < coordinates_dict[gene]['left']:
                    coordinates_dict[gene]['left'] = left
                    
                if right > coordinates_dict[gene]['right']:
                    coordinates_dict[gene]['right'] = right
                    
    gff_file.close()
                    
    mm5_file = open("C:/Gresham/tiny_projects/quorf/1-s2.0-S2405471220302404-mmc5_tab.txt")
    
    mm5_dict = {}
    
    for line in mm5_file:
        #ORF-RATER_orf-name	common-name	gene	chrom	gcoord	gstop	strand	start-codon	ORF-length	ORF-type	score_ORF-RATER	gene-description	annotated-ORF-length	extension-length	score_PhyloCSF	annotated AUG captured by ORF-RATER	evidence of translation at annotated AUG	include	category	Reference and notes
        if line.split('\t')[17] != 'FALSE':
            gene = line.split('\t')[2]
            if gene in coordinates_dict:
                mm5_dict[gene] = coordinates_dict[gene]
                ext_length = int(line.split('\t')[13])*3
                exp_start_codon = line.split('\t')[7]
                
                mm5_dict[gene]['ext_length'] = int(line.split('\t')[13])*3
                mm5_dict[gene]['exp_start_codon'] = line.split('\t')[7]
                
                if line.split('\t')[17] == 'TRUE':
                    mm5_dict[gene]['validated'] = 'predicted'
                else:
                    mm5_dict[gene]['validated'] = 'molecular'
                
            else:
                print(gene, 'hmmm...')
                
    mm5_file.close()
    
    source = 'eb_predicted'
    istype = 'NTE'
    is_uorf = 'True'
    
    for gene in mm5_dict:
        process = True
        
        chromo = mm5_dict[gene]['chromo']
        sign = mm5_dict[gene]['sign']
        ext_length = mm5_dict[gene]['ext_length']
        
        exp_start_codon = mm5_dict[gene]['exp_start_codon']
        left = mm5_dict[gene]['left']
        right = mm5_dict[gene]['right']
        
        validated = mm5_dict[gene]['validated']
        
        if sign == '+':
            
            _sc, valid_orf = pull_for_start(chromo, left, sign, 'ATG')
            
            if valid_orf:
                left = left - ext_length
                nte_start, start_match = pull_for_start(chromo, left, sign, exp_start_codon)
                
                exp_stop_codon, stop_nt = scan_for_stop(chromo, left, sign)
                stop_codon, stop_match = pull_for_stop(chromo, stop_nt, sign, exp_stop_codon)
                
                name = ('{gene}_{start}').format(gene = gene, start = left)
                
                if not start_match:
                    if nte_start not in start_codons:
                        process = False
                        
                if not stop_match:
                    print("not nte", nte_start, left, valid_orf, gene, mm5_dict[gene])
                    print(exp_stop_codon, stop_codon)
                    process = False                
            else:
                print("not valid_orf", gene)
                process = False
                
        if sign == '-':
            
            _sc, valid_orf = pull_for_start(chromo, right, sign, 'ATG')
            
            if valid_orf:
                right = right + ext_length
                nte_start, start_match = pull_for_start(chromo, right, sign, exp_start_codon)
                
                exp_stop_codon, stop_nt = scan_for_stop(chromo, right, sign)
                stop_codon, stop_match = pull_for_stop(chromo, stop_nt + 1, sign, exp_stop_codon)
                
                name = ('{gene}_{start}').format(gene = gene, start = right)
                
                if not start_match:
                    if nte_start not in start_codons:
                        process = False
                        
                if not stop_match:
                    print("not nte", nte_start, right, valid_orf, gene, mm5_dict[gene])
                    print(exp_stop_codon, stop_codon)
                    process = False                
            else:
                print("not valid_orf", gene)
                process = False
                
        if process:
            
            deets = (chromo, istype, left, right, sign, 
                name, is_uorf, gene, source,
                start_codon, exp_start_codon, start_match,
                stop_codon, exp_stop_codon, stop_match,
                validated)
            
            if sign == '+':
                left_mod = 0
                right_mod = 0
            
            if sign == '-':
                left_mod = 0
                right_mod = 1
                
            output_to_gff(uncorrected_outfile_name, left_mod, right_mod, deets)
            output_to_gff(validated_outfile_name, left_mod, right_mod, deets, True)
    

'''
'''
def parse_qc_sn(line):

    _ct, gene, left, right, sign, is_uorf, source, validated, = line.split(',')
    
    chr_alphabet = gene[1]
    chromo = alphabet_to_chromo[chr_alphabet]
            
    exp_start_codon = 'NNN'
    exp_stop_codon = 'NNN'
    istype = '_uORF'
    
    left = int(left)
    right = int(right)
    
    if sign == '+':
        start = left + 1
        stop = right + 1
        
    if sign == '-':
        start = right
        stop = left + 1
            
    name = ('{gene}_{start}').format(gene = gene, start = start)
                
    start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
    stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
            
    deets = (chromo, istype, left, right, sign, 
        name, is_uorf, gene, source,
        start_codon, exp_start_codon, start_match,
        stop_codon, exp_stop_codon, stop_match,
        validated)
    
    if start_match == True and stop_match == True:
        if start_codon not in start_codons:
            print('sn _uncorrected start_codon')
            print(line)
            print(right, left, start, start_codon, stop_codon)
            1/0
        
        if stop_codon not in stop_codons:
            print('sn _uncorrected stop_codon')
            print(line)
            print(right, left, stop, start_codon, stop_codon)
            1/0
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
            
        output_to_gff(uncorrected_outfile_name, left_mod, right_mod, deets)
        output_to_gff(validated_outfile_name, left_mod, right_mod, deets, True)
    else:
        print(' not (start_match == True and stop_match == True)')
        1/0
    ''' no evidence of these found 12.07.24                    
    if start_match == False or stop_match == False:            
        if start_codon not in start_codons:
            print('sn no_match start_codon')
            print(line)
            print(right, left, start, start_codon)
            1/0
            
        if stop_codon not in stop_codons:
            print('sn no_match stop_codon')
            print(line)
            print(right, left, stop, stop_codon)
            1/0
            
        #output precorrection
        #output_to_gff(mismatch_outfile_name, 0, 0, deets)
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
                
        left = left + left_mod
        right = right + right_mod
        
        if sign == '+':
            stop_codon, right = scan_for_stop(chromo, left, sign)
        
        if sign == '-':
            stop_codon, left = scan_for_stop(chromo, right, sign)
            
        if sign == '+':
            start = left
            stop = right
            
        if sign == '-':
            start = right
            stop = left + 1
                
        start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
        stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
        
        if (start_codon not in start_codons) or (not start_match):
            print('sn corrected start_codon')
            print(line)
            print(right, left, start, start_codon, exp_start_codon)
            1/0
            
        if (stop_codon not in stop_codons) or (not stop_match):
            print('sn corrected stop_codon')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon) 
            1/0
        
        deets = (chromo, istype, left, right, sign, 
            name, is_uorf, gene, source,
            start_codon, exp_start_codon, start_match,
            stop_codon, exp_stop_codon, stop_match,
            validated)
        
        if sign == '+':
            left_mod = 0
            right_mod = -1
        
        if sign == '-':
            left_mod = 0
            right_mod = -1
            print('There should be none of these - 11.07.2024')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon)
            1/0
        
        #output_to_gff(corrected_outfile_name, left_mod, right_mod, deets)
        #output_to_gff(validated_outfile_name, left_mod, right_mod, deets)
        '''

def parse_qc_sn_predicted(line):

    _ct, gene, left, right, sign, is_uorf, source, validated, = line.split(',')
    
    chr_alphabet = gene[1]
    chromo = alphabet_to_chromo[chr_alphabet]
            
    exp_start_codon = 'NNN'
    exp_stop_codon = 'NNN'
    istype = '_uORF'
    
    left = int(left)
    right = int(right)
    
    if '-' in gene:
        sign_check = gene.split('-')[0]
    else:
        sign_check = gene
        
    if check_name_to_sign[sign_check[-1]] != sign:
        sign = check_name_to_sign[sign_check[-1]]
    
    if sign == '+':
        start = left + 1
        stop = right + 1
        
    if sign == '-':
        start = right
        stop = left + 1
            
    name = ('{gene}_{start}').format(gene = gene, start = start)
                
    start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
    stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
            
    deets = (chromo, istype, left, right, sign, 
        name, is_uorf, gene, source,
        start_codon, exp_start_codon, start_match,
        stop_codon, exp_stop_codon, stop_match,
        validated)
    
    if start_match == True and stop_match == True:
        # No evidence 12.07.24
        # if start_codon not in start_codons:
        #     print('sn_predicted _uncorrected start_codon')
        #     print(line)
        #     print(right, left, start, start_codon, stop_codon)
        
        if stop_codon not in stop_codons:            
            if sign == '+':
                stop_codon, right = scan_for_stop(chromo, left+1, sign)
                right -= 1
            
            if sign == '-':
                stop_codon, left = scan_for_stop(chromo, right, sign)
                
            deets = (chromo, istype, left, right, sign, 
                name, is_uorf, gene, source,
                start_codon, exp_start_codon, start_match,
                stop_codon, exp_stop_codon, stop_match,
                validated)
            
            if stop_codon not in stop_codons:
                print('sn_predicted still_uncorrected stop_codon')
                print(line)
                print(right, left, stop, start_codon, stop_codon)
                1/0
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
            
        output_to_gff(uncorrected_outfile_name, left_mod, right_mod, deets)
        output_to_gff(validated_outfile_name, left_mod, right_mod, deets, True)
    else:
        print(' not (start_match == True and stop_match == True)')
        1/0
        
    ''' no evidence of these found 12.07.24                    
    if start_match == False or stop_match == False:            
        if start_codon not in start_codons:
            print('sn no_match start_codon')
            print(line)
            print(right, left, start, start_codon)
            1/0
            
        if stop_codon not in stop_codons:
            print('sn no_match stop_codon')
            print(line)
            print(right, left, stop, stop_codon)
            1/0
            
        #output precorrection
        #output_to_gff(mismatch_outfile_name, 0, 0, deets)
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
                
        left = left + left_mod
        right = right + right_mod
        
        if sign == '+':
            stop_codon, right = scan_for_stop(chromo, left, sign)
        
        if sign == '-':
            stop_codon, left = scan_for_stop(chromo, right, sign)
            
        if sign == '+':
            start = left
            stop = right
            
        if sign == '-':
            start = right
            stop = left + 1
                
        start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
        stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
        
        if (start_codon not in start_codons) or (not start_match):
            print('sn corrected start_codon')
            print(line)
            print(right, left, start, start_codon, exp_start_codon)
            1/0
            
        if (stop_codon not in stop_codons) or (not stop_match):
            print('sn corrected stop_codon')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon) 
            1/0
        
        deets = (chromo, istype, left, right, sign, 
            name, is_uorf, gene, source,
            start_codon, exp_start_codon, start_match,
            stop_codon, exp_stop_codon, stop_match,
            validated)
        
        if sign == '+':
            left_mod = 0
            right_mod = -1
        
        if sign == '-':
            left_mod = 0
            right_mod = -1
            print('There should be none of these - 11.07.2024')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon)
            1/0
        
        #output_to_gff(corrected_outfile_name, left_mod, right_mod, deets)
        #output_to_gff(validated_outfile_name, left_mod, right_mod, deets)
        '''

def parse_qc_eb_validated(line):

    _ct, gene, left, right, sign, is_uorf, source, validated, = line.split(',')
    
    chr_alphabet = gene[1]
    chromo = alphabet_to_chromo[chr_alphabet]
            
    exp_start_codon = 'NNN'
    exp_stop_codon = 'NNN'
    istype = '_uORF'
    
    if right == 'oORF':
        istype = 'oORF'
        left = int(left)
        stop_codon, right = scan_for_stop(chromo, left+1, sign)
        right -= 1
        
    if left == 'oORF':
        istype = 'oORF'
        stop_codon, left = scan_for_stop(chromo, right, sign)
    
    left = int(left)
    right = int(right)
    
    if sign == '+':
        if (istype != 'oORF'):
            right += 3
        start = left + 1
        stop = right + 1
        
    if sign == '-':
        # if (left != 'oORF'):
        #     left -= 3
        start = right
        stop = left + 1
            
    name = ('{gene}_{start}').format(gene = gene, start = start)
                
    start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
    stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
            
    deets = (chromo, istype, left, right, sign, 
        name, is_uorf, gene, source,
        start_codon, exp_start_codon, start_match,
        stop_codon, exp_stop_codon, stop_match,
        validated)
    
    if start_match == True and stop_match == True:
        if start_codon not in start_codons:
            print('eb_validated start_codon')
            print(line)
            print(right, left, start, start_codon, stop_codon)
            #1/0
        
        if stop_codon not in stop_codons:
            print('eb_validated stop_codon')
            print(line)
            print(right, left, stop, start_codon, stop_codon)
            #1/0
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
            
        print(line)
        output_to_gff(uncorrected_outfile_name, left_mod, right_mod, deets)
        output_to_gff(validated_outfile_name, left_mod, right_mod, deets, True)
    else:
        print(' not (start_match == True and stop_match == True)')
        1/0
        
    '''
    #no evidence of these found 12.07.24                    
    if start_match == False or stop_match == False:
        print('this?')
        1/0
            
        if start_codon not in start_codons:
            print('sn no_match start_codon')
            print(line)
            print(right, left, start, start_codon)
            1/0
            
        if stop_codon not in stop_codons:
            print('sn no_match stop_codon')
            print(line)
            print(right, left, stop, stop_codon)
            1/0
            
        #output precorrection
        #output_to_gff(mismatch_outfile_name, 0, 0, deets)
        
        if sign == '+':
            left_mod = 1
            right_mod = 0
        
        if sign == '-':
            left_mod = 1
            right_mod = 0
                
        left = left + left_mod
        right = right + right_mod
        
        if sign == '+':
            stop_codon, right = scan_for_stop(chromo, left, sign)
        
        if sign == '-':
            stop_codon, left = scan_for_stop(chromo, right, sign)
            
        if sign == '+':
            start = left
            stop = right
            
        if sign == '-':
            start = right
            stop = left + 1
                
        start_codon, start_match = pull_for_start(chromo, start, sign, exp_start_codon)
        stop_codon, stop_match = pull_for_stop(chromo, stop, sign, exp_stop_codon)
        
        if (start_codon not in start_codons) or (not start_match):
            print('sn corrected start_codon')
            print(line)
            print(right, left, start, start_codon, exp_start_codon)
            1/0
            
        if (stop_codon not in stop_codons) or (not stop_match):
            print('sn corrected stop_codon')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon) 
            1/0
        
        deets = (chromo, istype, left, right, sign, 
            name, is_uorf, gene, source,
            start_codon, exp_start_codon, start_match,
            stop_codon, exp_stop_codon, stop_match,
            validated)
        
        if sign == '+':
            left_mod = 0
            right_mod = -1
        
        if sign == '-':
            left_mod = 0
            right_mod = -1
            print('There should be none of these - 11.07.2024')
            print(line)
            print(right, left, stop, stop_codon, exp_stop_codon)
            1/0
        
        #output_to_gff(corrected_outfile_name, left_mod, right_mod, deets)
        #output_to_gff(validated_outfile_name, left_mod, right_mod, deets)
        '''

'''
'''

#convert to gff
may_2023_mismatch = set()

infile = open('C:/Gresham/tiny_projects/quorf/demo/validated_uorfs_v1.1.csv')

mismatch_outfile_name = ('C:/Gresham/tiny_projects/quorf/demo/_mismatch_uorfs_v1.2.gff')
mismatch_outfile = open(mismatch_outfile_name, 'w')
mismatch_outfile.close()

uncorrected_outfile_name = ('C:/Gresham/tiny_projects/quorf/demo/_uncorrected_uorfs_v1.2.gff')
uncorrected_outfile = open(uncorrected_outfile_name, 'w')
uncorrected_outfile.close()

corrected_outfile_name = ('C:/Gresham/tiny_projects/quorf/demo/_corrected_uorfs_v1.2.gff')
corrected_outfile = open(corrected_outfile_name, 'w')
corrected_outfile.close()

validated_outfile_name = ('C:/Gresham/tiny_projects/quorf/demo/validated_uorfs_v1.2_test.gff')
validated_outfile = open(validated_outfile_name, 'w')
validated_outfile.close()
#outside_name_set = set()

for line in infile:
    if line.split(',')[0] != '':
        line = line.strip()
        source = line.split(',')[6]

        if source == 'may_2023':
            parse_qc_may_2023(line)
            
        if (source == 'sn_standard') or (source == 'sn_expanded'):
            parse_qc_sn(line)
                
        if (source == 'sn_predicted'):
            parse_qc_sn_predicted(line)
            
        if (source == 'eb_validated'):
            parse_qc_eb_validated(line)
            
#v1.2
uncorrected_outfile_name = ('C:/Gresham/tiny_projects/quorf/demo/_uncorrected_uorfs_v1.2.gff')
uncorrected_outfile = open(uncorrected_outfile_name, 'w')
uncorrected_outfile.close()

parse_qc_eisenberg_2020()