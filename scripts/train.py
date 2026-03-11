import os
import sys
import numpy as np 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.eegnet import EEGNet
from data.loader import load_subject_data
from training.trainer import train

subjects = range(1, 10)
all_histories = []
accuracies = []

for s in subjects:

    X_train, y_train = load_subject_data(f'data/BCICIV_2a_gdf/A0{s}T.gdf')
    X_test, y_test = load_subject_data(f'data/BCICIV_2a_gdf/A0{s}E.gdf',
                                        label_file_path=f'data/BCICIV_2a_gdf/A0{s}E.mat')
    print(f"Subject: {s}")
    model = EEGNet()
    acc, history = train(model, X_train, y_train, X_test, y_test)
    accuracies.append(acc)
    all_histories.append({'subject': s, 'history': history})

print(f'Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
print(all_histories)
from utils.plot_results import plot_results
plot_results(all_histories, accuracies)

