# Confidence Calibration in Vision-Language-Action Models

This repository contains the code for the paper ***Confidence Calibration in Vision-Language-Action Models*** by Thomas Zollo and Richard Zemel, published in Transactions on Machine Learning Research (TMLR).

Paper Link: https://arxiv.org/abs/2507.17383

# Setup

    pip install -e .

# Calibration Experiments

The code for our calibration experiments is contained in the notebooks folder.

 - **main_exp.ipynb**: Code for experiment 1
 - **ens_exp.ipynb**: Code for experiment 2
 - **reprompt_ablation_{1/2}.ipynb**: Ablations for experiment 2
 - **across_time.ipynb**: Code for experiment 3
 - **recalibration.ipynb**: Code for experiment 4

# Producing Outputs for Calibration Experiments

To produce the data for our experiments, run each model in the LIBERO environment. For each episode, save a list with the output data from each timestep:

    timestep_output_data = {
        "logits": logits,
        "probs": probs,
        "predicted_token_ids": predicted_token_ids,
    }

Save data to:

    ../results/{model_name}/{cfg.task_suite_name}/{prompt_key}

where prompt_key corresponds to whether estimates are produced with the original instruction or a rephrasing.

The code for producing instruction rephrasings can be found in **build_reprompt_dataset.ipynb**.


# Citation

    @misc{zollo2025confidencecalibrationvisionlanguageactionmodels,
        title={Confidence Calibration in Vision-Language-Action Models},
        author={Thomas P Zollo and Richard Zemel},
        year={2025},
        eprint={2507.17383},
        archivePrefix={arXiv},
        primaryClass={cs.RO},
        url={https://arxiv.org/abs/2507.17383},
    }
