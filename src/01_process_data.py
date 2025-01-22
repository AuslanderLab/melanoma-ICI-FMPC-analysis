import pandas as pd
import glob
import sys

cpathd = sys.argv[1]
cnm = sys.argv[2]
cspec = sys.argv[3]
cdata_in = sys.argv[4]
cread_count = sys.argv[5]

# print(cpathd + ' ' + cnm + ' ' + cspec + ' ' + cdata_in + ' ' + cread_count)

col_names = ["id", "prot", "pident", "length","b","c","d","e","f","g","evalue","i"]

def process_blast_out(pathd,nm,spec):
    cdhit = pd.read_csv('data/supp_data_3.csv',index_col='id')
    clst = list(set(cdhit['clstr']))
    fls = glob.glob(pathd+spec)
    cdict = {cdhit.index[i]:cdhit['clstr'][i] for i in range(len(cdhit))}
    dall = [];xid=[]
    for file in fls:
        print(file)
        cnts = {clst[i]: 0 for i in range(len(clst))}
        x = pd.read_csv(file, sep='\t', header=None, names=col_names)
        x2 = x[(x.pident > 80) & (x.length > 20) & (x.evalue < 0.00000001)]
        xid.append(file.split('/')[-1].split('.txt')[0].split('.fastq.')[0].split('_')[0])
        x2.index = [cdict[i] for i in x2['prot']]
        uc = list(set(x2.index))
        for p in uc:
            dj = x2.loc[p]
            cnts[p] = len(set(dj['id']))
        dall.append(cnts)
    df = pd.DataFrame(dall,index = xid,columns=clst)
    df2 = df.T
    df2=df2.groupby(df2.columns, axis=1).sum()
    df2.to_csv(nm+'.csv')

def normalize_data(data_in,read_count):
    data = pd.read_csv(data_in, index_col=0)
    data.columns = [i.replace('blastx-', '') for i in data.columns]
    numreads = pd.read_csv(read_count, index_col='id')
    cols = list(data.index)
    data_out=data.copy()
    for c in cols:
        data_out.loc[c] = data.loc[c] / numreads['TotalReads'] * 1000000
    data_out.to_csv(data_in.split('.csv')[0]+'_norm.csv')


process_blast_out(cpathd, cnm, cspec)
normalize_data(cdata_in, cread_count)

# not run
# process_blast_out($BLASTX_OUTPUT, $FILE_PREFIX, spec='*.fastq.txt')
