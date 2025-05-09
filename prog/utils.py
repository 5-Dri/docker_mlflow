import mlflow
import numpy as np

import torch
from torch_geometric.utils import subgraph

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def log_params_from_omegaconf_dict(params):
    for param_name, value in params.items():
        print('{}: {}'.format(param_name, value))
        mlflow.log_param(param_name, value)

def log_artifacts(artifacts):
    if artifacts is not None:
        for artifact_name, artifact in artifacts.items():
            if artifact is not None:
                np.save(artifact_name, artifact.to('cpu').detach().numpy().copy())
                mlflow.log_artifact(artifact_name)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels), correct