import re
import os 
from Bio import Phylo
from io import StringIO
import subprocess
import time
import random
from multiprocessing import Pool
import numpy as np 
import math
import pandas
import scipy.stats as st

TEST = False
FIX_LEN = True
FIX_LEN_TEST = False

# output_dir = "data_gen/fixed_len_data_randomlambda_taxa50_lenmix_2000_gtr+i+g"

output_dir = "./empiral_GTR+I+G_1000_test"
excuting_dir = "iqtree-2.2.2.6-Linux/bin/iqtree2"
evo_model = "GTR+I+G"
# selected_species = [20, 50, 100] 
selected_species = [50]
# num_alignments = 16
num_alignments = 1000

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print(f"Folder '{output_dir}' created.")

# copy generate.py to output_dir
subprocess.run(f"cp generate.py {output_dir}", shell=True, check=True)

empirical_GTR_param_path = "./GTRparam.csv" #"/home/dingshizhe/NeuralNJ/empiricalGTRdist-master/GTRparam.csv"

class Distribution(object):
    
    def __init__(self,dist_names_list = []):
        self.dist_names = ['alpha', 'beta', 'bradford', 'chi', 'chi2', 'dgamma', 'dweibull', 'erlang', 'exponnorm', 'exponweib', 
                           'exponpow', 'gamma', 'genlogistic', 'genpareto', 'gennorm', 'genexpon', 'gengamma', 'halflogistic', 
                           'halfnorm', 'halfgennorm', 'invgamma', 'invgauss', 'invweibull', 'laplace', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 
                           'maxwell', 'norm', 'pareto', 'powerlaw', 'powerlognorm', 'powernorm', 'uniform', 'weibull_max', 'weibull_min']
        self.dist_results = []
        self.params = {}
        
        self.DistributionName = ""
        self.PValue = 0
        self.Param = None
        
        self.isFitted = False
        
    def Fit(self, y):
        # y = np.asarray(y)
        if np.all(y == y[0]) or np.any(np.isnan(y)):
            raise ValueError("输入数据全为常数或包含NaN，无法拟合分布！")
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(st, dist_name)
            try:
                param = dist.fit(y)
                self.params[dist_name] = param
                # Applying the Kolmogorov-Smirnov test
                D, p = st.kstest(y, dist_name, args=param)
                self.dist_results.append((dist_name, p))
            except Exception as e:
                # 拟合失败则跳过
                continue

        if not self.dist_results:
            raise RuntimeError("所有分布拟合均失败，请检查输入数据！")

        # select the best fitted distribution
        sel_dist, p = max(self.dist_results, key=lambda item: item[1])
        # store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.Param = self.params[sel_dist]
        self.PValue = p

        self.isFitted = True
        return self.DistributionName, self.PValue, self.Param

def empirical_dist(df):
    AC = [float(i) for i in df['A-C'].values]
    AG = [float(i) for i in df['A-G'].values]
    AT = [float(i) for i in df['A-T'].values]
    CG = [float(i) for i in df['C-G'].values]
    CT = [float(i) for i in df['C-T'].values]
    GT = [float(i) for i in df['G-T'].values]
    A = [float(i) for i in df['A'].values]
    C = [float(i) for i in df['C'].values]
    G = [float(i) for i in df['G'].values]
    T = [float(i) for i in df['T'].values]

    Gamma = [float(i) for i in df['Gamma'].values]
    Invar = [float(i) for i in df['Invar'].values]

    
    return [AC, AG, AT, CG, CT, GT, A, C, G, T, Gamma, Invar]


def bestFit_paramDist(paramDist):
    parameters = {'A-C':paramDist[0], 'A-G':paramDist[1], 'A-T':paramDist[2], 'C-G':paramDist[3], 'C-T':paramDist[4], 'G-T':paramDist[5], 'A':paramDist[6], 'C':paramDist[7], 'G':paramDist[8], 'T':paramDist[9], 'Gamma':paramDist[10], 'Invar':paramDist[11]}
    fitted_dist = {}
    for parameter, dist in parameters.items():
        dst = Distribution()
        fitted_dist[parameter] = dst.Fit(dist)
    return fitted_dist

paramDist = empirical_dist(pandas.read_csv(empirical_GTR_param_path))
paramProbabilityDist = bestFit_paramDist(paramDist)


def rand_GTR_dist(paramProbabilityDist):
    params = {}
    for parameter in ['A-C', 'A-G', 'A-T', 'C-G', 'C-T', 'G-T']:
        dist = paramProbabilityDist[parameter]
        dis = getattr(st, dist[0])
        value = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)
        while value <= 0 or value > 100:
            value = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)
        params[parameter] = value

    for parameter in ['A', 'C', 'G', 'T']:
        dist = paramProbabilityDist[parameter]
        dis = getattr(st, dist[0])
        value = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)
        while value < 0:
            value = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)
        params[parameter] = value

    freqSum = params['A'] + params['C'] + params['G'] + params['T']
    piA = params['A'] / freqSum
    piC = params['C'] / freqSum
    piG = params['G'] / freqSum
    piT = params['T'] / freqSum

    dist = paramProbabilityDist['Gamma']
    dis = getattr(st, dist[0])
    gamma = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)
    while gamma <= 0.1 or gamma > 10:
        gamma = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)

    dist = paramProbabilityDist['Invar']
    dis = getattr(st, dist[0])
    invar = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)
    while invar < 0 or invar > 1:
        invar = round(dis.rvs(*dist[2][:-2], loc=dist[2][-2], scale=dist[2][-1], size=1)[0], 5)

    Q = np.array([
        params['A-C'], params['A-G'], params['A-T'],
        params['C-G'], params['C-T'], params['G-T'],
        piA, piC, piG, piT, gamma, invar
    ])
    return Q


def generate_random_tree(leaf_count, branch_length=1.0):
    tree = Phylo.BaseTree.Tree.randomized(leaf_count, branch_length=1.0)
    root_node = tree.root
    root_node.name = ""
    root_node.branch_length = None  
    
    for node in tree.get_nonterminals():
        log2 = math.log(2.0)
        log5 = math.log(5.0)
        _lambda = math.exp( random.uniform(log2, log5) )
        node.branch_length = random.expovariate(_lambda)
        node.name = ""
    
    for node in tree.get_terminals():
        # node.branch_length = random.expovariate(1.0 / 0.1)
        log2 = math.log(2.0)
        log5 = math.log(5.0)
        _lambda = math.exp( random.uniform(log2, log5) )
        node.branch_length = random.expovariate(_lambda)

    tree_handle = StringIO()
    Phylo.write(tree, tree_handle, 'newick')

    newick_tree = tree_handle.getvalue()
    tree_handle.close()

    pattern = r'(:\d+\.\d+;)$'
    match = re.search(pattern, newick_tree)

    if match:
        modified_newick_tree = newick_tree.replace(match.group(1), ";")
        #print(modified_newick_tree)
    return modified_newick_tree


def generate_align_seq(tree_str, leaf_count=50, site_len=2000,insertion_rate = 0.0,deletion_rate = 0.1,num_alignments=1,output_dir="/home/zxr/phylogenetic_tree/INDELibleV1.03/generate/data_gen", number = 0):
    label = "G"
    # leaf_count = 2  

    output_filename = f"{label}_l_{site_len}_n_{leaf_count}_{insertion_rate}_{deletion_rate}_{number}"

    # Save the tree to a file with .tre extension
    with open(f"{output_dir}/{output_filename}_raw.tre", "w") as tree_file:
        tree_file.write(tree_str)
    
    Q = rand_GTR_dist(paramProbabilityDist)
    AC, AG, AT, CG, CT, GT, piA, piC, piG, piT, gamma, invar = Q
    # model_str = f"GTR{{{AC},{AG},{AT},{CG},{CT},{GT},{A},{C},{G},{T}}}+I{{{invar}}}+G{{{gamma}}}"
    model_str = (
        f"GTR{{{AC},{AG},{AT},{CG},{CT}}}"
        f"+F{{{piA},{piC},{piG},{piT}}}"
        f"+I{{{invar}}}"
        f"+G{{{gamma}}}"
    )
    command = [
        excuting_dir,
        "--alisim", f"{output_dir}/{output_filename}",
        "-m", model_str,
        "-t", f"{output_dir}/{output_filename}_raw.tre",
        "--length", str(site_len),
        "--indel", f"{insertion_rate},{deletion_rate}",
        "--no-unaligned",
        "--seed", f"{number}",
    ]

    command_str = " ".join(command)


    try:
        subprocess.run(command_str, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"generate_control failed: {e}")

    return f"{output_dir}/{output_filename}.phy",output_filename


def process_length_site(len_list, selected_species=selected_species , num_alignments=num_alignments):
    for length in len_list:
        len_out_path = f"{output_dir}/len{length}"
        if not os.path.exists(len_out_path):
            os.mkdir(len_out_path)
            print(f"Folder '{len_out_path}' created.")
        else:
            print(f"Folder '{len_out_path}' already exists.")

        for species in selected_species:
            taxa_len_out_path = f"{len_out_path}/taxa{species}"
            if not os.path.exists(taxa_len_out_path):
                os.mkdir(taxa_len_out_path)
                print(f"Folder '{taxa_len_out_path}' created.")
            else:
                print(f"Folder '{taxa_len_out_path}' already exists.")

            insertion_rate = 0
            for i in range(num_alignments):
                deletion_rate = random.choice([0, 0.01, 0.02, 0.03, 0.04])
                tree_str = generate_random_tree(species)
                generate_align_seq(tree_str, leaf_count=species, site_len=length, insertion_rate=insertion_rate, deletion_rate=deletion_rate, num_alignments=1, output_dir=taxa_len_out_path, number=i)
            print(f"species: {species}, len: {length}")


# 使用 multiprocessing
if __name__ == '__main__':
    # site_lens = [ [i for i in range(128, 1024+1, 32)] ]
    site_lens = [[ 256, 512, 1024]]  
    # site_lens = [[2048, 4096]]
    # selected_species = [[ 100]] 
    #num_alignments = 1024
    #args = (site_lens, selected_species, num_alignments) 
    
    with Pool() as pool:
        pool.map(process_length_site, site_lens)
