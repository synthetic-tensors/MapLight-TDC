# ADMET properties From MapLight TDC 

The following models have weights in the weights folder. Regression models also have a scaler applied which is pickled and must be used to unscale each model's inference.

An example of making a prediction using the loaded model weights and scaler for each checkpoint:
```
model = cb.CatBoostRegressor(**params)
model.load_model(os.path.join('saved_weights',f"{name}"))
with open('saved_weights/{name}_scaler.pk','rb') as f:
    Y_scaler = pickle.load(f)
y_pred_test = Y_scaler.inverse_transform(model.predict(X_test)).reshape(-1)
```

Classification models have a threshold. The optimal threshold for maximum balanced accuracy is given below in metrics. AN example of using it for a given checkpoint is:

```
model = cb.CatBoostClassifier(**params)
model.load_model(os.path.join('saved_weights',f"{name}"))
y_pred_test = model.predict_proba(X_test)[:,1]
return (y_pred_test>optimal_th).astype(int)
```

## Regression Models:
|                           |       mae |   spearman |
|---------------------------|-----------|------------|
| caco2_wang                |  0.27223  |   0.85793  |
| lipophilicity_astrazeneca |  0.516466 |   0.831155 |
| solubility_aqsoldb        |  0.800037 |   0.866275 |
| ppbr_az                   |  7.47501  |   0.712983 |
| vdss_lombardo             |  1.68696  |   0.717743 |
| half_life_obach           |  7.45942  |   0.558608 |
| clearance_hepatocyte_az   | 30.2953   |   0.502324 |
| clearance_microsome_az    | 19.954    |   0.653661 |
| ld50_zhu                  |  0.629854 |   0.609439 |

## Classification Models:
|                                         |     aupr |    auroc |   balacc |   optimal_th |
|-----------------------------------------|----------|----------|----------|--------------|
| bioavailability_ma                      | 0.901531 | 0.727303 | 0.699577 |   0.828497   |
| hia_hou                                 | 0.996875 | 0.989712 | 0.938856 |   0.977044   |
| pgp_broccatelli                         | 0.941559 | 0.938816 | 0.880885 |   0.414728   |
| bbb_martins                             | 0.963123 | 0.913618 | 0.84018  |   0.865201   |
| cyp2c9_veith                            | 0.859974 | 0.931277 | 0.860052 |   0.353022   |
| cyp2d6_veith                            | 0.791154 | 0.924813 | 0.839032 |   0.16205    |
| cyp3a4_veith                            | 0.915578 | 0.931526 | 0.858056 |   0.403832   |
| cyp2c9_substrate_carbonmangels          | 0.432612 | 0.654368 | 0.670762 |   0.14914    |
| cyp2d6_substrate_carbonmangels          | 0.709706 | 0.802199 | 0.742048 |   0.289062   |
| cyp3a4_substrate_carbonmangels          | 0.68598  | 0.641501 | 0.653162 |   0.534332   |
| herg                                    | 0.949307 | 0.881443 | 0.813093 |   0.689034   |
| ames                                    | 0.906939 | 0.869584 | 0.758416 |   0.488999   |
| dili                                    | 0.903321 | 0.909565 | 0.783887 |   0.528811   |
| tox21-ar-bla-agonist-p1                 | 0.68322  | 0.946455 | 0.833223 |   0.0318758  |
| tox21-ar-bla-antagonist-p1              | 0.659154 | 0.907511 | 0.811298 |   0.120367   |
| tox21-elg1-luc-agonist-p1               | 0.383235 | 0.830848 | 0.730952 |   0.0202452  |
| tox21-er-bla-agonist-p2                 | 0.610997 | 0.897829 | 0.800731 |   0.0706903  |
| tox21-er-bla-antagonist-p1              | 0.469869 | 0.889797 | 0.80674  |   0.0881795  |
| tox21-erb-bla-antagonist-p1             | 0.686544 | 0.911833 | 0.829754 |   0.126843   |
| tox21-erb-bla-p1                        | 0.591014 | 0.908448 | 0.847735 |   0.0194611  |
| tox21-err-p1_agonist                    | 0.545362 | 0.912336 | 0.848086 |   0.0293485  |
| tox21-fxr-bla-agonist-p2                | 0.118553 | 0.828313 | 0.708267 |   0.019386   |
| tox21-fxr-bla-antagonist-p1             | 0.624882 | 0.940786 | 0.872408 |   0.0697105  |
| tox21-pgc-err-p1_agonist                | 0.54195  | 0.915476 | 0.837401 |   0.0291077  |
| tox21-ppard-bla-agonist-p1              | 0.194658 | 0.89306  | 0.840848 |   0.0189974  |
| tox21-ppard-bla-antagonist-p1           | 0.55482  | 0.900251 | 0.841596 |   0.0503388  |
| tox21-pparg-bla-agonist-p1              | 0.253157 | 0.777719 | 0.700157 |   0.0411015  |
| tox21-pparg-bla-antagonist-p1           | 0.553884 | 0.892889 | 0.809238 |   0.078308   |
| tox21-pr-bla-agonist-p1                 | 0.79252  | 0.907427 | 0.880763 |   0.0200623  |
| tox21-pr-bla-antagonist-p1              | 0.732251 | 0.920156 | 0.844547 |   0.16993    |
| tox21-rt-viability-hek293-p1_glo 8 hr   | 0.603564 | 0.92003  | 0.852772 |   0.102452   |
| tox21-rt-viability-hek293-p1_glo 40 hr  | 0.660304 | 0.896794 | 0.837891 |   0.142036   |
| tox21-rt-viability-hek293-p1_flor 16 hr | 0.378712 | 0.825721 | 0.792897 |   0.0308674  |
| tox21-rt-viability-hek293-p1_flor 24 hr | 0.493268 | 0.833737 | 0.768325 |   0.0488585  |
| tox21-rt-viability-hek293-p1_flor 32 hr | 0.511791 | 0.850615 | 0.779082 |   0.0599788  |
| tox21-rt-viability-hek293-p1_flor 40 hr | 0.492634 | 0.845957 | 0.760244 |   0.0812956  |
| tox21-rt-viability-hek293-p1_flor 0 hr  | 0.128291 | 0.74392  | 0.549118 |   0.0197711  |
| tox21-rt-viability-hek293-p1_flor 8 hr  | 0.4032   | 0.797755 | 0.772996 |   0.0298919  |
| tox21-rt-viability-hek293-p1_glo 0 hr   | 0.328627 | 0.860411 | 0.790658 |   0.0096723  |
| tox21-rt-viability-hek293-p1_glo 16 hr  | 0.604654 | 0.89483  | 0.815798 |   0.108697   |
| tox21-rt-viability-hek293-p1_glo 24 hr  | 0.676462 | 0.890974 | 0.833918 |   0.120451   |
| tox21-rt-viability-hek293-p1_glo 32 hr  | 0.674068 | 0.899056 | 0.831646 |   0.131387   |
| tox21-rxr-bla-agonist-p1                | 0.283895 | 0.768877 | 0.668002 |   0.0369999  |
| tox21-sbe-bla-agonist-p1                | 0.368894 | 0.975641 | 0.695192 |   0.0096832  |
| tox21-sbe-bla-antagonist-p1             | 0.607067 | 0.907174 | 0.852524 |   0.100327   |
| tox21-trhr-hek293-p1_agonist            | 0.226899 | 0.778906 | 0.72887  |   0.00980396 |
| tox21-trhr-hek293-p1_antagonist         | 0.30715  | 0.913514 | 0.812196 |   0.00931424 |
| tox21-tshr-agonist-p1                   | 0.327101 | 0.871387 | 0.755946 |   0.0482734  |
| tox21-tshr-antagonist-p1                | 0.343284 | 0.883851 | 0.826512 |   0.0288489  |
| tox21-tshr-wt-p1                        | 0.667834 | 0.696517 | 0.831734 |   0.00969448 |
| tox21-vdr-bla-agonist-p1                | 0.379164 | 0.869008 | 0.640051 |   0.0128356  |
| tox21-vdr-bla-antagonist-p1             | 0.506907 | 0.932286 | 0.873052 |   0.0596334  |

[![arXiv](https://img.shields.io/badge/arXiv-2310.00174-b31b1b.svg)](https://arxiv.org/abs/2310.00174)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maplightrx/MapLight-TDC/blob/main/submission.ipynb)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)

This repository contains the source code for MapLight's [Therapeutics Data Commons (TDC) ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) submission.

## Installation

This codebase describes MapLight's two submissions to the TDC leaderboards:

1. MapLight model ([`submission.ipynb`](https://github.com/maplightrx/MapLight-TDC/blob/main/submission.ipynb)): [CatBoost](https://catboost.ai/) gradient boosted decision trees with [ECFP](https://pubmed.ncbi.nlm.nih.gov/20426451/), [Avalon](https://pubmed.ncbi.nlm.nih.gov/16995723/), and [ErG fingerprints](https://pubmed.ncbi.nlm.nih.gov/16426057/), as well as [200 physicochemical descriptors](https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/). Runnable on Colab.

2. MapLight + GNN model ([`submission_gnn.ipynb`](https://github.com/maplightrx/MapLight-TDC/blob/main/submission_gnn.ipynb)): the same as the MapLight model with [graph isomorphism network (GIN) supervised masking fingerprints](https://arxiv.org/abs/1905.12265) from [`molfeat`](https://molfeat.datamol.io/featurizers/gin_supervised_masking). __WARNING__: Not runnable on Colab becuase of [this issue](https://github.com/datamol-io/molfeat/issues/61).

Both notebooks will install all dependencies in a new Python environment with [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html):

```
# Create an environment for this project
mamba create -n maplight python=3.10 -y && mamba activate maplight

mamba install jupyterlab
```

