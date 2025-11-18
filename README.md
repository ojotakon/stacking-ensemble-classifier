# Manual Stacking Ensemble Classifier

This repository contains a clean implementation of a manual stacking ensemble classifier
using scikit-learn. The meta-model is trained on out-of-fold (OOF) predictions generated
by multiple base learners, without using `sklearn.ensemble.StackingClassifier`.

## Project Structure
- `Stacking_demo.ipynb` — Interactive notebook with step-by-step outputs  
- `stacking.py` — Clean Python script version  
- `stacking_results.png` — Example output (OOF folds + final accuracy summary)

## Models Used
- Decision Tree (base learner)  
- SVM (base learner)  
- Logistic Regression (meta-learner)  
- Soft Voting Classifier (baseline comparison)

## Key Features
- Manual generation of out-of-fold (OOF) predictions  
- Training of a meta-learner using stacked base-model outputs  
- Performance comparison: stacking vs. individual models vs. soft voting  
- Reproducible pipeline suitable for learning, research, or small-scale deployment  

## How to Run
To execute the script version:

python stacking.py

The notebook can be run directly using Google Colab or Jupyter Notebook.

## Output
The script prints:
- Fold-by-fold training logs  
- OOF prediction progress  
- Final accuracy scores for:  
  - Stacking ensemble  
  - Voting ensemble  
  - Base models  

Refer to `stacking_results.png` for an example output snapshot.
