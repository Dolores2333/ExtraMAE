# Instructions for Reimplementation 

We provide the official implementation of Time Series Generation with Masked Autoencoder under the folder **MAI**. 

You may implement experiments on Stock, Sine, and Energy according to the ***README.md*** under **MAI**. 

We also provide three jupyter notes for you to preliminarily check the results.

- *ExtraMAE.ipynb* shows experiments on visualization, discrimination scores, prediction scores, and ablation study. 
- *Imputaition.ipynb* shows results for imputation
- *MaskRatio.ipynb* shows results in the study of mask ratios. 

 The jupyter notes rely on large experiment results. Therefore, the readers cannot run them directly. You may find the links for all results of experiments and runnable jupyter notes in *SupplementaryMaterial.pdf*. 
 
 # Masked Autoencoder with Extrapolator (ExtraMAE) 

## Intro

The folder **MAI** provides the official implementation of Time Series Generation with Masked Autoencoder. 

We summarize the prediction, discrimination, and visualization results in summary.ipynb. You may refer to it directly or reimplement our experiments by the steps below. 

## Repository Structure

#### Model

Pytorch implementation of ExtraMAE can be found in ```./MAI/models.py```. All supporting files are stored in ```./MAI/modules/*```.

modules include: 

- ```utils.py``` stores all utility functions 
- ```generation.py``` generates masks and synthetic data
- ```visualization.py``` visualize the original data and synthetic data by PCA plots, t-SNE plots, and dot plots. 

#### Datasets

Datasets are stored in ```./MAI/data/*```

#### Metrics

Metrics for assessing the quality of synthetic data are in ```./MAI/metrics/*```:

- Predictive Scores: We train a predictor on the synthetic data and do testing on the original data. Mean absolute value (MAE) on the original data is reported as the predictive score. (The lower, the better. )
- Discriminative Scores: We train a discriminator to tell synthetic data from the original data. Accuracy on the testing set is reported as the discriminative score. (The lower, the better. )

#### Configuration

```config_stock.json``` configs for the stock dataset. 

```config_energy.json``` configs for energy dataset. 

```config_sine.json``` configs for sine dataset. 

```requirements.txt``` stores all packages needed.

#### Results

A experiment on Stock named ```AE0_EM0_RE50000``` , for example, has a private folder ```./MAI/storage/AE0_EM0_RE50000``` for its own and there are three subfolders under it. 

- ```model``` stores model related files
- ```pics ``` stores visualization for the evaluation of ExtraMAE by generation mode ```random_once```
- ```synthesis``` stores visualization for three other generation modes:
  - ```cross_average```
  - ```cross_concate```
  - ```random_average```

```summary.ipynb``` summarizes the results. 

## Implementation

1. Create a conda environment named MAI.

   ```python
   conda create -n MAI python=3.6
   conda activate MAI
   ```

2. Enter the folder of the repository MAI in the terminal. 

3. Get your environment ready.

   ```pip install -r requirements.txt```

4. Run main.py with the config you need.

   ```python main.py --config_dir=stock_config.json```
   
   ```python main.py --config_dir=energy_config.json```
   
   ```python main.py --config_dir=sine_config.json```

5. In case you wanna change the configuration for different experiments and instances, just modify the config.json and save it. 

6. To see the summarized results of predictive scores and discriminative scores, send the ```experiment_name``` to method ```find_results``` in ```MAI/summary.ipynb```.


