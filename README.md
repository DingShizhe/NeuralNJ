# NeuralNJ

Code implementation for "Neural Neighbor Join for Accurate Molecular Phylogenetic Inference".

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

## Generating Synthetic Data

To generate synthetic data with MSA lengths from 128 to 1024 and 50 species, navigate to the `data_gen` directory and run:

```bash
cd data_gen/
python generate.py
```

## Real Data

For real data, refer to the work described in the article [Harnessing machine learning to guide phylogenetic-tree search algorithms](https://osf.io/b8aqj/).

## Training

To train the model using real data, use the following command:

```bash
python train.py --config_path config/pretrain_real.yaml
```

To train the model using synthetic data, use the following command:

```bash
python train.py --config_path config/pretrain_mix.yaml
```

## Reinforced Search

To perform reinforced search, run:

```bash
python finetune_rl_search.py --config_path config/finetune_reinforce_search.yaml
```

## Direct Inference

To perform direct inference, modify the `infer_opt` option in `finetune_reinforce_search.yaml` to either "Argmax" or "Search", and then run:

```bash
python finetune_rl_search.py --config_path config/finetune_reinforce_search.yaml
```
