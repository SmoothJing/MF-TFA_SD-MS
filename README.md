# MF-TFA_SD-MS
## Introduction
The official implementation of "A Singing Melody Extraction Network Via Self-Distillation and Multi-Level Supervision."

We propose a singing melody extraction network
consisting of five stacked multi-scale feature time-frequency ag-
gregation (MF-TFA) modules. In the same network, deeper layers
generally contain more contextual information than shallower
layers. To help the shallower layers enhance the ability of task-
relevant feature extraction, we propose a self-distillation and
multi-level supervision (SD-MS) method, which leverages the fea-
ture distillation from the deepest layer to the shallower one and
multi-level supervision to guide network training. 

## Getting Started

### Download Datasets

- [MIR-1k](https://sites.google.com/site/sites/system/errors/WebspaceNotFound?path=%2Funvoicedsoundseparation%2Fmir-1k)
- [ADC 2004 & MIREX05](https://labrosa.ee.columbia.edu/projects/melody/)
- [MEDLEY DB](https://medleydb.weebly.com/)

## Important updata
The entire code scripts will be made public after being licensed.
We expect to release full code and training details in December.

## Special thanks

- [Haojie Wei](https://github.com/Dream-High)
- [Ke Chen](https://github.com/KnutKeChen)
- [Shuai Yu](https://github.com/yushuai)
