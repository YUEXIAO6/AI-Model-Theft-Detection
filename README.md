# A Management Framework for AI Service Security: Detecting Stealthy Model Extraction

This repository contains the official implementation, simulation environment, and evaluation scripts for our paper on detecting stealthy AI model extraction attacks via multi-scale traffic analysis. 

Our framework leverages Spatial Pyramid Pooling (SPP) combined with application-layer semantic proxies to identify highly adaptive, low-frequency, and distributed model extraction attacks against AI APIs.

##  Key Contributions
- Custom API Simulation: A high-fidelity, highly imbalanced API traffic generator simulating stealthy model extraction (replacing standard CV-based frameworks to focus purely on network/L7 behavior).
- Semantic SPP Framework: Fusing application-layer semantic proxies (payload diversity, endpoint entropy) with Spatial Pyramid Pooling to capture multi-scale temporal dynamics.
- Strict Evaluation: Entity-disjoint splits, strict 1.0% Normal-FPR constraints, and operational metrics (Alerts/Hr) designed for real-world Security Operations Centers (SOCs).

##  Repository Structure
```text
AI-Model-Theft-Detection/
│
├── README.md                                    
│
├── 1_simulation_env/               # API Servers and Traffic Generators
│   ├── credit_api/                 # Base scenario (Credit Scoring API)
│   ├── insurance_api/              # Cross-API generalization (4D parameter space)
│   └── concept_drift/              # Temporal concept drift scenario
│
├── 2_feature_extraction/           # Core Logic
│   ├── spp_experiments.py          # Feature extraction & SPP context building
│   └── window_relabel_credit_only.py # High-fidelity window alignment & labeling
│
├── 3_evaluation/                   # Experimental Results Reproduction
│   ├── robustness_eval_entity_split.py  # Main robustness table
│   ├── robustness_eval_ablation.py      # Ablation study
│   ├── robustness_eval_baserate.py      # Base rate sensitivity (PPV/Precision)
│   ├── robustness_eval_drift.py         # Concept drift evaluation
│   ├── robustness_eval_insurance.py     # Cross-API evaluation
│   ├── dl_baselines.py                  # GRU+Attention baselines
│   └── dl_baselines_gridsearch.py       # DL Hyperparameter tuning
│
└── 4_analysis_and_plots/           # Visualization Scripts
    ├── log_driven_surrogate.py          # Surrogate model learning curve
    ├── sensitivity_analysis.py          # W and K parameter sensitivity
    ├── spp_information_gain.py          # Mutual Information (MI) analysis
    └── plot_temporal_distribution.py    # Traffic distribution visualization
```
## Installation
We recommend using Python 3.8 or higher.
```bash 
pip install fastapi uvicorn pandas numpy scikit-learn xgboost torch matplotlib requests
```

##  Reproducibility Guide
 IMPORTANT NOTE: Please run ALL commands from the root directory of this repository. This ensures that the generated traffic_logs.csv is saved in the correct location and cross-directory imports work properly.

Step 1: Generate Simulation Traffic (Credit API)
We need to generate the dataset by simulating normal users and various stealthy attackers.
Open Terminal 1 and start the target API server:  
```bash 
python 1_simulation_env/credit_api/server.py
```
Open Terminal 2 and run the traffic generator: 
```bash
python 1_simulation_env/credit_api/traffic_generator_imbalanced.py
```
(Note: Let the generator run for at least 1.5 to 2 hours to collect sufficient imbalanced data. The script will automatically stop and save traffic_logs.csv to the root directory).
     
Step 2: Evaluate Detection Robustness
Once traffic_logs.csv is generated, you can reproduce the main tables from the paper.
Run the main entity-disjoint robustness evaluation:
```bash 
python 3_evaluation/robustness_eval_entity_split.py
```
(Note: Running this script will also generate a detection_records.json file, which is required for the learning curve analysis in Step 3).
Run the comprehensive ablation study:
```bash 
python 3_evaluation/robustness_eval_ablation.py
```
Run Deep Learning Baselines (GRU + Attention): 
```bash 
python 3_evaluation/dl_baselines.py
```

Step 3: Analysis and Plotting
Generate the visualizations used in the paper.
1. Surrogate Model Learning Curve:
Demonstrates how the SPP framework truncates the attacker's model extraction process before they achieve high fidelity.
```bash 
python 4_analysis_and_plots/log_driven_surrogate.py
```
2. Parameter Sensitivity (W & K): 
```bash 
python 4_analysis_and_plots/sensitivity_analysis.py
```
3. Temporal Traffic Distribution: 
```bash 
python 4_analysis_and_plots/plot_temporal_distribution.py
```

##  Advanced Scenarios
A. Concept Drift Evaluation
To evaluate the framework's robustness against temporal behavior drift (Phase 1 to Phase 2):
# Terminal 1
```bash 
python 1_simulation_env/concept_drift/server_drift.py
```

# Terminal 2 (Run for 2 hours)
```bash 
python 1_simulation_env/concept_drift/traffic_generator_drift.py
```

# After completion, evaluate:
```bash 
python 3_evaluation/robustness_eval_drift.py
```

B. Cross-API Generalization (Insurance Pricing)
To evaluate the framework on a different API with a higher-dimensional (4D) parameter space:
# Terminal 1
```bash 
python 1_simulation_env/insurance_api/server_insurance.py
```

# Terminal 2 (Run for 1.5 hours)
```bash 
python 1_simulation_env/insurance_api/traffic_generator_insurance.py
```

# After completion, evaluate:
```bash 
python 3_evaluation/robustness_eval_insurance.py
```

## Copyright & Contact

Copyright (c) 2026 Yuelin Chen. All rights reserved.

This repository contains the source code for our manuscript: *"A Management Framework for AI Service Security: Detecting Stealthy Model Extraction via Multi-Scale Traffic Analysis"*

- The code is provided for academic and research purposes.
- If you have any questions regarding the code or the paper, please feel free to open an issue or contact the corresponding author at: yuelin.chen@connect.polyu.hk.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.