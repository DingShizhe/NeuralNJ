import os
import subprocess
import time

excuting_dir = "raxml-ng/build/bin/raxml-ng"
evo_model = "GTR+I+G"
# source_dir = "data_gen/fixed_len_128_data_n_30_100_8K"
source_dir = "data_gen/fixed_len_data_randomlambda_taxa50_lenmix_2000_gtr+i+g"
output_dir = "data_gen/CSolver/fixed_len_data_randomlambda_taxa50_lenmix_2000_gtr+i+g"

# taxanum = 110
length_list = [i for i in range(128, 1024+1, 32)]

def command_raxml(msa_path):
    # get dir and file name of msa_path
    dir_name = os.path.dirname(msa_path)
    file_name = os.path.basename(msa_path)
    output_paht = os.path.join(dir_name, file_name)

    command = [
        excuting_dir,
        "--msa", f"{msa_path}",
        "--model", evo_model,
        "--threads", "8",
        "--prefix", f"{output_paht}",
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

# copy phy from data to CSolver
def copy_phy_to_CSolver(taxanum=40, len=1000):
    data_dir = f"{source_dir}/len{len}/taxa{taxanum}"
    
    # if len{len} dir not exit in CSolver , create it
    CSolver_dir = f"{output_dir}/len{len}/taxa{taxanum}"
    if not os.path.exists(CSolver_dir):
        os.makedirs(CSolver_dir)
        print(f"Folder '{CSolver_dir}' created.")

    for file in os.listdir(data_dir):
        if file.endswith(".phy"):
            file_path = os.path.join(data_dir, file)
            subprocess.run(f"cp {file_path} {CSolver_dir}", shell=True, check=True)

# copy tre file fome source to CSolver as _raw.tre
def copy_tre_to_CSolver(taxanum=40, len=1000):
    data_dir = f"{source_dir}/len{len}/taxa{taxanum}"
    
    # if len{len} dir not exit in CSolver , create it
    CSolver_dir = f"{output_dir}/len{len}/taxa{taxanum}"
    # if not os.path.exists(CSolver_dir):
    #     os.makedirs(CSolver_dir)
    #     print(f"Folder '{CSolver_dir}' created.")

    for file in os.listdir(data_dir):
        if file.endswith("_raw.tre"):
            file_path = os.path.join(data_dir, file)
            # first copy the file as _raw.tre in data_dir, then copy _raw.tre to CSolver
            file_prefix = file
            dis_file_path = f"{data_dir}/{file_prefix}"
            subprocess.run(f"cp {file_path} {dis_file_path}", shell=True, check=True)
            subprocess.run(f"cp {dis_file_path} {CSolver_dir}", shell=True, check=True)




# file end with ".phy.raxml.bestTree" change to ".tre"
def change_tre_name(taxanum=40, len=1000):
    CSolver_dir = f"{output_dir}/len{len}/taxa{taxanum}"
    for file in os.listdir(CSolver_dir):
        if file.endswith(".phy.raxml.bestTree"):
            file_path = os.path.join(CSolver_dir, file)
            new_file_path = file_path.replace(".phy.raxml.bestTree", ".tre")
            subprocess.run(f"mv {file_path} {new_file_path}", shell=True, check=True)

# list all the files in the output_dir and command raxml and change the name
def command_raxml_and_change_name(taxanum=40, len=1000):
    CSolver_dir = f"{output_dir}/len{len}/taxa{taxanum}"
    count = 0
    for file in os.listdir(CSolver_dir):
        if file.endswith(".phy"):
            file_path = os.path.join(CSolver_dir, file)
            count += command_raxml(file_path)
    change_tre_name(taxanum, len)
    return count

import glob

def rename_and_delete_files(directory):
    # Iterate over all files ending with ".phy.raxml.log" and rename them
    for file_path in glob.glob(os.path.join(directory, "*.phy.raxml.log")):
        new_file_path = file_path.replace(".phy.raxml.log", ".log")
        os.rename(file_path, new_file_path)
        print(f"Renamed '{file_path}' to '{new_file_path}'")

    # Delete all files that match "*.phy.raxml*"
    for file_path in glob.glob(os.path.join(directory, "*.phy.raxml*")):
        os.remove(file_path)
        print(f"Deleted '{file_path}'")


for length in length_list:
    for i in [50]:
        #record the time of running the command
        taxanum = i
        copy_phy_to_CSolver(taxanum, length)
        copy_tre_to_CSolver(taxanum, length)
        start = time.time()
        count = command_raxml_and_change_name(taxanum=taxanum, len=length)
        end = time.time()
        avg_time = (end - start) / count

        rename_and_delete_files(f"{output_dir}/len{length}/taxa{taxanum}")
        print(f"taxanum: {taxanum}, len: {length}, count: {count}, time: {end - start}, avg_time: {avg_time}")

        # taxanum = i
        # copy_tre_to_CSolver(taxanum, length)
        # print("taxanum, length: ", taxanum, length, "DONE...")