import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

mode = ['spot', 'hline', 'vline']

print('[INFO]Creating Heatmap..')
model_path = 'models/checkpoints/18Feb/'
output_path = 'outputs/Patch/After Adversarial Training'
for model_name in os.listdir(model_path):
    for mode_i in mode:
        file_name = ['meanlogit', 'meanprobability', 'count', 'mse']
        for file_name_i in file_name:
            df = pd.read_csv(
                f'{output_path}/{model_name}/{mode_i}-{file_name_i}.csv')
            plt.figure(figsize=[14, 12])

            if file_name_i == 'meanlogit':
                df = df.pivot('Original Label', 'Perturbed Label', 'Mean Logit')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='g', vmin=0.0, vmax=1.0)
                # sns.heatmap(hline_meanlogit, linewidths=0.8, annot=True, cmap='RdBu', vmin=0.0, vmax=1.0)
            if file_name_i == 'meanprobability':
                df = df.pivot('Original Label', 'Perturbed Label', 'Mean Probability')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='g', vmin=0.0, vmax=100.0)
            if file_name_i == 'count':
                df = df.pivot('Original Label', 'Perturbed Label', 'Count')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='g', annot_kws={"fontsize":12})
            if file_name_i == 'mse':
                df = df.pivot('Original Label', 'Perturbed Label', 'Average MSE')
                sns.heatmap(df, linewidths=0.8, annot=True, fmt='.0f', annot_kws={"fontsize":12})

            print(f'Saving: {output_path}/{model_name}/{mode_i}-{file_name_i}.png')
            plt.savefig(
                f'{output_path}/{model_name}/{mode_i}-{file_name_i}.png')

print('[INFO]Done')
