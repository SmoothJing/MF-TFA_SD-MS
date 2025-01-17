# MF-TFA_SD-MS

## Introduction
The official implementation of "A Singing Melody Extraction Network Via Self-Distillation and Multi-Level Supervision."
Our paper has been accepted by 2025 ICASSP.

We propose a singing melody extraction network
consisting of five stacked multi-scale feature time-frequency ag-
gregation (MF-TFA) modules. In the same network, deeper layers
generally contain more contextual information than shallower
layers. To help the shallower layers enhance the ability of task-
relevant feature extraction, we propose a self-distillation and
multi-level supervision (SD-MS) method, which leverages the fea-
ture distillation from the deepest layer to the shallower one and
multi-level supervision to guide network training. 

<img src="https://github.com/SmoothJing/MF-TFA_SD-MS/blob/main/fig/arch.png" alt="Table" width="1800">

## Getting Started

### Download Datasets

- [MIR-1k](https://sites.google.com/site/sites/system/errors/WebspaceNotFound?path=%2Funvoicedsoundseparation%2Fmir-1k)
- [ADC 2004 & MIREX05](https://labrosa.ee.columbia.edu/projects/melody/)
- [MEDLEY DB](https://medleydb.weebly.com/)

## Results

### Prediction result

The visualization illustrates that our proposed method can reduce the octave errors and the melody detection errors.

![image](https://github.com/SmoothJing/MF-TFA_SD-MS/blob/main/fig/visualization-2.png)

### Comprehensive result

The bold values indicate the best performance for a specific metric.

<img src="https://github.com/SmoothJing/MF-TFA_SD-MS/blob/main/fig/results-1.png" alt="Table" width="1800">

### Ablation study result_1

Results of ablation experiments introducing a self-distillation and multi-level supervision method in partially existing singing melody extraction model. SD-MS indicates that self-distillation and multi-level supervision is used.

<img src="https://github.com/SmoothJing/MF-TFA_SD-MS/blob/main/fig/results-2.png" alt="Table" width="1800">

### Ablation study result_2r

Ablation study of the loss function on three datasets

![image](https://github.com/SmoothJing/MF-TFA_SD-MS/blob/main/fig/results-s.png)

## Important updata

The entire code scripts will be made public after being licensed.

## Special thanks

- [Haojie Wei](https://github.com/Dream-High)
- [Ke Chen](https://github.com/KnutKeChen)
