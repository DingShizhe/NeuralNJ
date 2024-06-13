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

output_dir = "data_gen/fixed_len_data_randomlambda_taxa50_lenmix_2000"
excuting_dir = "iqtree-2.2.2.6-Linux/bin/iqtree2"
evo_model = "JC"
# selected_species = [20, 30, 40, 50, 100, 200] # 同上
selected_species = [50]
# num_alignments = 128  # 同上
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
    # 生成标号、叶子节点数、q、r、ir、dr
    label = "G"
    # leaf_count = 2  # 这里假设有两个叶子节点

    # 构建输出文件名
    output_filename = f"{label}_l_{site_len}_n_{leaf_count}_{insertion_rate}_{deletion_rate}_{number}"

    # Save the tree to a file with .tre extension
    with open(f"{output_dir}/{output_filename}.tre", "w") as tree_file:
        tree_file.write(tree_str)
    
        # 构建要执行的命令
    command = [
        excuting_dir,
        "--alisim", f"{output_dir}/{output_filename}",
        "-m", evo_model,
        "-t", f"{output_dir}/{output_filename}.tre",
        "--length", str(site_len),
        "--indel", f"{insertion_rate},{deletion_rate}",
        "--no-unaligned",
        "--seed", f"{number}",
        # "--num-alignments", str(num_alignments)
    ]

    # 将命令列表转换为字符串
    command_str = " ".join(command)


    try:
        subprocess.run(command_str, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"generate_control 命令执行失败: {e}")

    
    
    return f"{output_dir}/{output_filename}.phy",output_filename


def process_length_site(len_list, selected_species=selected_species , num_alignments=num_alignments):
    for length in len_list:
        len_out_path = f"{output_dir}/len{length}"
        if not os.path.exists(len_out_path):
            os.mkdir(len_out_path)
            print(f"Folder '{len_out_path}' created.")
        else:
            print(f"Folder '{len_out_path}' already exists.")

        # selected_species = random.sample(range(min_species, max_species + 1), num_species)
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


# min_species = 3
# max_species = 100

# min_species = 30
# max_species = 100
# if TEST:
#     num_species = 4
#     num_alignments = 32
# if FIX_LEN:
#     num_species = 71
#     num_alignments = 1024

# min_length_site = 100
# max_length_site = 10000

# site_lens = [i for i in range(100,10000)]
# for len in site_lens:
#     len_out_path = f"{output_dir}/len{len}"
#     if not os.path.exists(len_out_path):
#         os.mkdir(len_out_path)
#         print(f"Folder '{len_out_path}' created.")
#     else:
#         print(f"Folder '{len_out_path}' already exists.")

#     selected_species = random.sample(range(min_species, max_species + 1), num_species)
#     for species in selected_species:
#         taxa_len_out_path = f"{len_out_path}/taxa{species}"
#         if not os.path.exists(taxa_len_out_path):
#             os.mkdir(taxa_len_out_path)
#             print(f"Folder '{taxa_len_out_path}' created.")
#         else:
#             print(f"Folder '{taxa_len_out_path}' already exists.")

#         insertion_rate = 0.0 
#         deletion_rate = round(random.uniform(0.01, 0.11), 2)
#         for i in range(num_alignments):
#             tree_str = generate_random_tree(species)
#             generate_align_seq(tree_str, leaf_count=species, site_len=len, insertion_rate = 0.0,deletion_rate = deletion_rate,num_alignments=1,output_dir=taxa_len_out_path,number=i)
#         print(f"species: {species}, len: {len}")



# 使用 multiprocessing
if __name__ == '__main__':
    site_lens = [ [i for i in range(128, 1024+1, 32)] ]  # 注意，这里是嵌套列表，因为我们将单个参数作为列表传递
    #selected_species = [[20, 30, 40, 50, 100, 200]]  # 同上
    #num_alignments = 1024  # 同上
    #args = (site_lens, selected_species, num_alignments)  # 打包成一个参数
    
    with Pool() as pool:
        pool.map(process_length_site, site_lens)  # 注意这里传递的是一个列表的列表
