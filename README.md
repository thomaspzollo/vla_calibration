# Confidence Calibration in Vision-Language-Action Models

This repository contains the code for the paper ***Confidence Calibration in Vision-Language-Action Models*** by Thomas Zollo and Richard Zemel.

Paper Link: https://arxiv.org/abs/2507.17383

# Calibration Experiments

The code for our calibration experiments are contained in the notebooks folder.  

 - **main_exp.ipynb**: Code for experiments 1 and 2
 - **reprompt_ablation_{1/2}.ipynb**: Ablations for experiment 2
 - **across_time.ipynb**: Code for experiment 3
 - **scaling_w_temp.ipynb**: Code for experiment 4

# Producing Outputs for Calibration Experiments

To produce the data for our experiments, run OpenVLA in the LIBERO environment.  For each episode, save a list with the output data from each timestep: 

    data_dict = {
        "logits": logits,
        "predicted_token_ids": predicted_token_ids,
    }

Save data to:

    ../results/{cfg.task_suite_name}/{prompt_key}

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