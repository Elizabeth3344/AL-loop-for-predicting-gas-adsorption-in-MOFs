# Active Learning Loop for Predicting Gas Adsorption in MOFs
Predicting CO₂ adsorption in Metal-Organic Frameworks (MOFs) using machine learning models (ALIGNN, Uni-MOF) with finetuning on GCMC simulation data from RASPA.

## Overview
This project implements an active learning pipeline for predicting CO₂ adsorption isotherms in MOFs:

1. **Screening**: Filter 20,372 MOFs from the QMOF database to 5,419 candidates suitable for CO₂ adsorption (PLD > 3.3 Å)
2. **ML Prediction**: Run inference with pretrained ALIGNN and Uni-MOF models on all candidates
3. **Selection**: Select top-10 MOFs by predicted uptake (ALIGNN at 2.5 bar, 298 K)
4. **GCMC Validation**: Compute accurate adsorption isotherms using RASPA for selected MOFs
5. **Finetuning**: Finetune both models on GCMC data and evaluate improvement on 10 random validation MOFs

> **Note:** CIF structure files are not included in this repository due to their large number (5,419 files). They can be downloaded from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF).

## Models

### ALIGNN
[Atomistic Line Graph Neural Network](https://github.com/usnistgov/alignn) — a GNN that builds two graphs from crystal structure: an atomic graph (atoms and distances) and a line graph (bonds and angles). Invariant to crystal symmetries through use of scalar distances and angles. Outputs 5 uptake values at fixed pressures (0.01, 0.05, 0.1, 0.5, 2.5 bar) at 298 K for CO₂.

### Uni-MOF
[Universal MOF Transformer](https://github.com/dptech-corp/Uni-MOF) — a Transformer-based model using self-attention mechanism. Accepts arbitrary gas, temperature, and pressure as input parameters. Outputs a single uptake value per query. Pretrained on 631K MOF/COF structures, finetuned on hMOF/MOFX-DB GCMC data.

## Data

- **QMOF Database**: 20,372 experimentally synthesized MOFs with DFT-optimized structures ([Rosen et al., 2021](https://doi.org/10.1016/j.matt.2021.02.015))
- **Filtering**: PLD > 3.3 Å (kinetic diameter of CO₂) → 5,419 MOFs
- **GCMC simulations**: Computed with [RASPA](https://github.com/iRASPA/RASPA2) at 298 K, pressures 0.01–10 bar

### Force Fields

GCMC simulations use parameters distributed with RASPA (LGPL license):
- **UFF** (Universal Force Field) — framework atoms ([Rappé et al., JACS, 1992](https://doi.org/10.1021/ja00051a040))
- **TraPPE** — CO₂ model ([Potoff & Siepmann, AIChE J., 2001](https://doi.org/10.1002/aic.690470719))
- **EQEq** - charge equilibration method for crystal structures, implemented in PyEQEq ([Ongari et al., JCTC, 2019](https://doi.org/10.1021/acs.jctc.8b00669); [GitHub](https://github.com/lsmo-epfl/pyeqeq))

## Metrics

All metrics computed on 10 validation MOFs (not seen during finetuning):

| Metric | Formula | Units |
|--------|---------|-------|
| MAE | (1/N) × Σ\|ŷᵢ − yᵢ\| | mol/kg |
| MSE | (1/N) × Σ(ŷᵢ − yᵢ)² | mol²/kg² |
| RMSE | √MSE | mol/kg |

## Requirements

### ALIGNN
```
Python 3.8+
alignn
jarvis-tools
dgl==1.1.3
torch
```

### Uni-MOF
```
Docker: dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
GPU with CUDA support
```

### RASPA + PyEQEq
```
RASPA2
Python 3.10 (required for pyeqeq compatibility)
Ubuntu 20.04 (required for pyeqeq compatibility)
pyeqeq
```

## Usage

### 1. Filter QMOF database
Filter 20,372 MOFs by pore limiting diameter (PLD > 3.3 Å) → 5,419 MOFs suitable for CO₂.
```bash
cd ML_Models/MOFs
python filterCO2_qmof.py
```

### 2. Run ALIGNN inference (before finetuning)
Predict CO₂ uptake for all 5,419 MOFs at 5 pressures (0.01–2.5 bar, 298 K).
```bash
cd ML_Models/ALIGNN
python batch_alignn.py
```

### 3. Select MOFs for finetuning and validation
Select top-10 MOFs by predicted uptake at 2.5 bar (train set), then 10 random MOFs excluding top-10 (validation set).
```bash
cd ML_Models/MOFs
python select_10MOFs.py
python select_random_10.py
```

### 4. Run GCMC in RASPA
Compute GCMC isotherms for all 20 selected MOFs (10 train + 10 valid) at 298 K, 0.01–10 bar.
```bash
cd GCMC
python main_experiment.py
```

### 5. Finetune ALIGNN
Finetune on 10 train MOFs with GCMC data, validate on 10 random MOFs, then run inference on all 5,419 MOFs and compute metrics.
```bash
cd ML_Models/ALIGNN
python prepare_finetune_data.py
python finetune_alignn.py
python batch_alignn_finetuned.py
python compute_metrics_alignn_v2.py
```

### 6. Finetune Uni-MOF (in Docker)
Same pipeline as ALIGNN but runs inside Docker container with GPU support.
```bash
docker run --rm -it --gpus all --shm-size=8g \
    -v $(pwd):/workspace \
    dptechnology/unimol:latest-pytorch1.11.0-cuda11.3 bash

cd /workspace/ML_Models/Uni-MOF
python make_finetune_lmdb_v2.py
bash run_finetune_v2.sh
bash run_infer.sh
python compute_metrics_v2.py
```

## References

- Rosen et al. "Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery." *Matter*, 2021. ([DOI](https://doi.org/10.1016/j.matt.2021.02.015))
- Choudhary & DeCost. "Atomistic Line Graph Neural Network for improved materials property predictions." *npj Computational Materials*, 2021. ([DOI](https://doi.org/10.1038/s41524-021-00650-1))
- Wang et al. "A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal-organic frameworks." *Nature Machine Intelligence*, 2024. ([DOI](https://doi.org/10.1038/s42256-024-00837-1))
- Dubbeldam et al. "RASPA: molecular simulation software for adsorption and diffusion in flexible nanoporous materials." *Molecular Simulation*, 2016. ([DOI](https://doi.org/10.1080/08927022.2015.1010082))
- Rappé et al. "UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations." *JACS*, 1992. ([DOI](https://doi.org/10.1021/ja00051a040))
- Potoff & Siepmann. "Vapor–liquid equilibria of mixtures containing alkanes, carbon dioxide, and nitrogen." *AIChE J.*, 2001. ([DOI](https://doi.org/10.1002/aic.690470719))

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Elizaveta Chernysheva
