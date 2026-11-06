# Instance-Optimal Estimation with Multiple LLM Judges on a Budget

Official implementation of **"Instance-Optimal Estimation with Multiple LLM Judges on a Budget"** (NeurIPS 2026).

This repository implements **EST-IVWE**, an adaptive allocation method for estimating response quality using multiple LLM judges with heterogeneous costs and variances.

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- Joblib

```bash
python -m pip install -r requirements.txt
```

## Repository Structure

- `synthetic_experiment.py`: Synthetic experiments comparing Uniform, Oracle, EST-IVWE (Bounded), and EST-IVWE (Gaussian).
- `real_experiment.py`: Experiments using repeated LLM-judge evaluations based on HelpSteer2.
- `cost_scenario.py`: Sensitivity analysis for different cost-variance relationships.
- `fig/`: Generated plots and experiment results.

Experiment parameters such as the budget, error norm, number of runs, and evaluated attribute can be configured in the execution block at the bottom of each script.
All scripts use a default random seed of `0`; change the `seed` argument to run a different reproducible trial.

## Running the Experiments

### Synthetic experiments

```bash
python synthetic_experiment.py
```

This reproduces the synthetic experiments for $K=1000$, $J=10$, and $p \in \{1,2,\infty\}$.

### Cost-sensitivity experiments

```bash
python cost_scenario.py
```

This evaluates the `Bad is Expensive`, `Bad is Cheap`, and `Uniform Costs` settings.

### Real-data experiments

```bash
python real_experiment.py
```

The real-data experiments require repeated LLM-judge evaluations derived from HelpSteer2.

## Data Availability

The original [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) dataset is publicly available under the CC BY 4.0 license.

The repeated LLM-judge evaluations used in this work were provided by Saha et al. and remain their assets. Therefore, these files are not redistributed in this repository.

For the data format and generation procedure, please refer to the repository accompanying [LLM-as-Judge on a Budget](https://github.com/Swagget/LLM-judgesteer).

Users with authorized access should place the JSONL files under:

```text
helpsteer2_neurips2026/
```

Provide one file for each combination of `complexity`, `correctness`, `helpfulness`, or `verbosity` and the three judges. Filenames must contain the corresponding attribute name and end in `.jsonl`.

Each record must contain at least:

```json
{"predicted_score": 3.0}
```

The synthetic experiments do not require any external data.

## Outputs

The scripts save plots and experiment results under:

```text
fig/synthetic/
fig/real/
```

Plots report the mean error over repeated runs with 10th-90th percentile shading.

## Citation

```bibtex
@inproceedings{lee2026instance,
  title     = {Instance-Optimal Estimation with Multiple LLM Judges on a Budget},
  author    = {Lee, Junghyun and Kim, Sanghwa and Jedra, Yassir and Prouti{\`e}re, Alexandre and Yun, Se-Young},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

We thank the authors of Saha et al. (2026) for providing the code and processed evaluation data used in the real-data experiments.
