# syntagen-sol

This is our solution for Syntagen Challenge.

## Installation

```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
python -m pip install -r requirements.txt
```

## Execution

Execution flow is figured as below:

![](./flow_visualization.png)

To execute the pipeline, start the `run.sh` file:

```bash
bash run.sh 
```

Artifacts will be stored in the `data` folder:
- `data/mask/*.png`: store all mask files of VOC annotation.
- `data/image/*.jpg`: store all generated images.