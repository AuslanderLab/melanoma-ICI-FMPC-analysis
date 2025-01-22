import glob
import pandas as pd

flatten = lambda l: [item for sublist in l for item in sublist]

def translate_frameshifted(sequence, gcode):
    """
    Translation of nucleotide to amino acid
    :param sequence: a section of nucleotide sequence
    :param gcode: gencode dictionary
    :return: amino acide sequence
    """
    translate = ''.join([gcode.get(sequence[3 * i:3 * i + 3]) for i in range(len(sequence) // 3)])
    return translate


def reverse_complement(sequence, bpairs):
    """
    Genertate the reversed sequence
    :param sequence: a section of nucleotide sequence
    :param bpairs: basepairs dictionary
    :return: the reversed string of the nucleotide sequence
    """
    reversed_sequence = (sequence[::-1])
    rc = ''.join([bpairs.get(reversed_sequence[i]) for i in range(len(sequence))])
    return rc


def six_frame_trans(seq, gcode, bpairs):
    x1 = translate_frameshifted(seq, gcode)
    x2 = translate_frameshifted(seq[1:], gcode)
    x3 = translate_frameshifted(seq[2:], gcode)
    rc = reverse_complement(seq, bpairs)
    x4 = translate_frameshifted(rc, gcode)
    x5 = translate_frameshifted(rc[1:], gcode)
    x6 = translate_frameshifted(rc[2:], gcode)
    x = [x1, x2, x3, x4, x5, x6]
    return x
 

gencode = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
    'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W'}

basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

files = glob.glob("*contigs.fasta.linear")

prot_seq = {}

for file in files:
    sample = file.split("/")[3].strip()
    print(sample)
    for line in open(file):
        line = line.strip()
        if not line.startswith(">"):
            sft = six_frame_trans(line, gencode, basepairs)
            sft_ = [i for i in sft if '_' not in i and len(i) >= 15] 
            if len(sft_) > 0:
                prot_seq[sample] = sft_ if sample not in prot_seq else prot_seq[sample] + sft_


join_seq = {}

for sample in prot_seq.keys():
    join_seq[sample] = '*'.join(prot_seq[sample])


# 15 overlapping kmers to find multi-patient patterns, and then 9-10 mers for the MHC prediction: 
k = 15

# get all kmers for patients
kmers = {}
for sample in join_seq.keys():
    # get all 15-mers that don't contain stops
    kmers[sample] = [j for j in [join_seq[sample][i:i+k] for i in range(len(join_seq[sample])-(k-1))] if "*" not in j]


kmer_freq = {}
j = 0
# get counts of each kmer in the dict
for i in set(flatten(list(kmers.values()))):
    count = 0
    j += 1
    for sample in kmers.keys():
        if i in kmers[sample]:
            count += 1
    if count > 12:
        kmer_freq[i] = count
    if j % 500 == 0:
        print(j)

