# Ensemble vs Ensemble 

[![My Skills](https://skillicons.dev/icons?i=python,pytorch)](https://skillicons.dev)   

![Alt text](https://github.com/Adversarial-Panda/ensemble_vs_ensemble/blob/main/images/ensemble_vs_ensemble_2.png) 


### Datasets 🗃
| Name        | Classes | 
|-------------|---------|
| ImageNet    | 1000    | 
| CIFAR-100   | 100     | 
| PAD-UFES-20 | 6       |  

---  

### Ensemble Attacks ⚔️
| Method   | Venue       | Year | Title                                                                 |
|----------|------------|------|-----------------------------------------------------------------------|
| Ens. Attack | ICLR       | 2017 | Delving into Transferable Adversarial Examples and Black-Box Attacks |
| SVRE     | CVPR       | 2022 | Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability |
| AdaEA    | CVPR       | 2023 | An Adaptive Model Ensemble Adversarial Attack for Boosting Adversarial Transferability |
| SMER     | CVPR       | 2024 | Ensemble Diversity Facilitates Adversarial Transferability           |
| NAMEA    | AAAI       | 2026 | Boosting Adversarial Transferability via Ensemble Non-Attention      |

### Pool of Ensemble Attacks 
| Surrogate Model | Type | Name on Timm              |
|-----------------|------|---------------------------|
| ResNet-18       | CNN  | resnet18                  |
| Inception-V3    | CNN  | inception_v3              |
| DeiT-Tiny       | ViT  | deit_tiny_patch16_224     |
| ViT-Tiny        | ViT  | vit_tiny_patch16_224      |

--- 

### Ensemble Models 🛡
|No  | Method                      | Abbreviation | Source    | Paper | 
|----|-----------------------------|--------------|-----------|-------| 
| 1  | Majority Voting (MV) - Soft | MV-S         | Benchmark | None  | 
| 2  | Majority Voting (MV) - Hard | MV-H         | Benchmark | None  |
| 3  | MV - Soft - Weighted        | W-MV-S       | Benchmark | None  |
| 4  | MV - Hard - Weighted        | W-MV-H       | Benchmark | None  |
| 5  | Stacking                    | Stack        | Benchmark | None  |
| 6  | Early Fusion                | EF           | Benchmark | None  |
| N  | Dynamic Uncertainty-based Selection | DUS  | Engineering Applications of Artificial Intelligence (2024) | None  |
| N  | Class-specified DEL         | CP-DEL       | Knowledge-Based Systems (2025)| None | 
