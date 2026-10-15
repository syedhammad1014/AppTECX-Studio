#!/bin/bash
set -e
python data/generate_data.py
python week1_python_basics/python_basics.py
python week2_data_cleaning/cleaning.py
python week3_eda_visualization/eda.py
python week4_regression/regression.py
python week5_trees_features/trees_features.py
python week6_svm_knn_nb_clustering/models_and_clustering.py
python week7_model_evaluation/evaluation.py
python week8_final_project/train.py
python week8_final_project/predict.py
