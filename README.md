# NeuralNJ

Code implementation for "Accurate and efficient phylogenetic inference through end-to-end deep learning".

## Environment Setup

To set up the environment, run the following commands:

```bash
conda env create -f environment.yaml
```

Note: `raxmlpy` needs to be installed separately. Please refer to the instructions in the `RAxMLpy` folder.

Activate the environment with:

```bash
conda activate NeuralNJ
```

## Third-Party Dependencies

- [IQTree](http://www.iqtree.org/): A tool for simulating evolutionary processes and generating MSA.
- [RAxML](https://cme.h-its.org/exelixis/web/software/raxml/): A solver for phylogenetic inference.


## Generating Synthetic Data

To generate synthetic data with MSA lengths from 128 to 1024 and 50 species, navigate to the `data_gen` directory and run:

```bash
cd data_gen/
# Make sure to set the `excuting_dir` variable in `generate.py` to the path of the IQTree executable
python generate.py
```

**Note: Pre-generated datasets under the GTR+G model used in the paper are provided in the `./data_gen/data` directory for training, validation, and testing.**

## Real Data

For real data, refer to the work described in the article [Evaluating Fast Maximum Likelihood-Based Phylogenetic Programs Using Empirical Phylogenomic Data Sets](https://doi.org/10.1093/molbev/msx302).

## Training

**To use the datasets provided in `./data_gen/data` for training or testing, please modify the data directory paths in the corresponding YAML configuration files.**

To train the model using synthetic data under the evolution model GTR+I+G, use the following command:

```bash
python train.py --config_path config/pretrain_mix.yaml
```


You can monitor training and validation curves using TensorBoard with the following command:
```bash
tensorboard --logdir tb_logs/
```

## Inference

To perform inference for NeuralNJ, run:

```bash
python finetune_rl_search.py --config ./config/finetune_reinforce_search_example.yaml --infer_opt Argmax
```

```bash
python finetune_rl_search.py --config ./config/finetune_reinforce_search_example.yaml --infer_opt Search
```

```bash
python finetune_rl_search.py --config ./config/finetune_reinforce_search_example.yaml --infer_opt Finetune
```

## Example Cases

The example configuration file `./config/finetune_reinforce_search_example.yaml` uses test cases located in the `examples` folder. Specifically, two cases in `examples/len1024taxa50`:

- `G_l_1024_n_50_0.0_0.08_58.phy`: Used in the phylogenetic analysis case study in the paper
- `G_l_1024_n_50_0.0_0.08_19.phy`: Used in the topology construction process analysis

Additionally, examples/cal_rf_distance.py provides a utility to calculate Robinson-Foulds distance between two phylogenetic trees. Use it with python cal_rf_distance.py --reftree <reference_tree_file> --inftree <inferred_tree_file> to evaluate topological differences between trees.
