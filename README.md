# MAST6100-Assessment---Drug-prediction
Final assessment using a UCI dataset evaluating whether personality traits and demographic variables are accurate predictors of drug use, the main drug focus is cocaine

Research Question - Can personality traits and demographic variables accurately predict whether an individual is a user of a given drug?

UCI dataset link - https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified

Required R packages - dplyr, readr, caret, corrplot, ggplot2, pROC, randomForest, xgboost, keras
To run the analysis - source("R code/Analysis.R")

Folder structure (and contents of repository) - 
MAST6100 Assessment 2 - Haniyah Amir
├── README.md
├── Data/
│   └── Drug consumption dataset   
├── R code/
│   └── Analysis.R            
├── Report/
│   └── Drug consumption dataset report       
├── Slides/
│   └── Drug consumption slides
└── Outputs/
    ├── glm_roc_curve.png
    ├── rf_varimp.png
    ├── xgb_roc_curve.png
    └── nn_roc_curve.png
