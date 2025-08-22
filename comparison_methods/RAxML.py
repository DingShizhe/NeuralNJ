import os
import subprocess
import time

excuting_dir = "./raxml-ng"
evo_model = "GTR+G"


def command_raxml(msa_path, output_dir):
    # get dir and file name of msa_path
    dir_name = os.path.dirname(msa_path)
    file_name = os.path.basename(msa_path)

    output_path = os.path.join(output_dir, file_name)

    command = [
        excuting_dir,
        "--msa", f"{msa_path}",
        "--model", evo_model,
        "--threads", "8",
        "--prefix", f"{output_path}",
        "--redo",
        "--seed", "7788",
        "--opt-model", "off"
    ]

    command_str = " ".join(command)
    
    try:
        subprocess.run(command_str, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"generate_control failed: {e}")
        return False


# suffix = '_iqtree240_fixdeletion'
# input_dir = f"/home/dingshizhe/NeuralNJ/NeuralNJ/data_gen/gtr+i+g_taxa100_lenmix{suffix}"
input_dir = f"/home/dingshizhe/NeuralNJ/NeuralNJ-main/data_gen/data/test"


# lenth_list = [1024]
taxa_len_list = [(100, 1024),(100, 256) ,(100, 512), (20, 1024), (50, 1024)]
# taxa_len_list = [(100, 2048)]

for taxa, length in taxa_len_list:
    # for length in lenth_list:
    cur_input_dir = os.path.join(input_dir, 'len'+str(length)+'taxa'+str(taxa))
    data_name = ''.join(cur_input_dir.split('/')[-1])

    output_dir = f"raxml_output_test128_GTR+G/{data_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for file in os.listdir(cur_input_dir):
        if file.endswith(".phy"):
            msa_path = os.path.join(cur_input_dir, file)
            command_raxml(msa_path, output_dir)