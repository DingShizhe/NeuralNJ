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

TEST = False
FIX_LEN = True
FIX_LEN_TEST = False

output_dir = "data_gen/fixed_len_data_randomlambda_taxa50_lenmix_2000_gtr+i+g"
excuting_dir = "iqtree-2.2.2.6-Linux/bin/iqtree2"
evo_model = "GTR+I+G"
# selected_species = [20, 30, 40, 50, 100, 200] 
selected_species = [50]
# num_alignments = 128 
num_alignments = 80

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print(f"Folder '{output_dir}' created.")

# copy generate.py to output_dir
subprocess.run(f"cp generate.py {output_dir}", shell=True, check=True)


def generate_random_tree(leaf_count, branch_length=1.0):
    tree = Phylo.BaseTree.Tree.randomized(leaf_count, branch_length=1.0)
    root_node = tree.root
    root_node.name = ""
    root_node.branch_length = None  
    
    for node in tree.get_nonterminals():
        log2 = math.log(2.0)
        log50 = math.log(50)
        _lambda = math.exp( random.uniform(log2, log50) )
        node.branch_length = random.expovariate(_lambda)
        node.name = ""
    
    for node in tree.get_terminals():
        # node.branch_length = random.expovariate(1.0 / 0.1)
        log2 = math.log(2.0)
        log5 = math.log(5.0)
        _lambda = math.exp( random.uniform(log2, log50) )
        node.branch_length = random.expovariate(_lambda)

    tree_handle = StringIO()
    Phylo.write(tree, tree_handle, 'newick')

    newick_tree = tree_handle.getvalue()
    tree_handle.close()

    pattern = r'(:\d+\.\d+;)$'
    match = re.search(pattern, newick_tree)

    if match:
        # 找到匹配项后，将其删除
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
    
    command = [
        excuting_dir,
        "--alisim", f"{output_dir}/{output_filename}",
        "-m", evo_model,
        "-t", f"{output_dir}/{output_filename}_raw.tre",
        "--length", str(site_len),
        "--indel", f"{insertion_rate},{deletion_rate}",
        "--no-unaligned",
        "--seed", f"{number}",
        # "--num-alignments", str(num_alignments)
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

            insertion_rate = 0.0 
            deletion_rate = round(random.uniform(0.01, 0.11), 2)
            for i in range(num_alignments):
                tree_str = generate_random_tree(species)
                generate_align_seq(tree_str, leaf_count=species, site_len=length, insertion_rate=insertion_rate, deletion_rate=deletion_rate, num_alignments=1, output_dir=taxa_len_out_path, number=i)
            print(f"species: {species}, len: {length}")


# 使用 multiprocessing
if __name__ == '__main__':
    site_lens = [ [i for i in range(128, 1024+1, 32)] ]  
    #selected_species = [[20, 30, 40, 50, 100, 200]] 
    #num_alignments = 1024
    #args = (site_lens, selected_species, num_alignments) 
    
    with Pool() as pool:
        pool.map(process_length_site, site_lens)
