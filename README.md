# IPL Score Prediction using Deep Learning

Predicts the final total of an IPL innings from the live match state (venue, teams, batsman, bowler, runs, wickets and overs) using a feed-forward neural network built with TensorFlow / Keras.

## Dataset

- **File:** `ipl_data.csv` (IPL 2008-2017, ball-by-ball match states)
- **Size:** 76,014 rows, 15 columns, 617 matches, no missing values
- **Target:** `total` (final innings score, range 67 to 263, mean about 161)

| Used as input (9 features) | Dropped |
|---|---|
| bat_team, bowl_team, venue, batsman, bowler, runs, wickets, overs, striker | mid, date (identifiers); runs_last_5, wickets_last_5, non-striker (highly correlated) |

## Project Structure

```
.
├── IPL_Score_Prediction.ipynb   # Full notebook: EDA, preprocessing, model, evaluation, prediction
├── ipl_data.csv                 # Dataset
├── requirements.txt             # Python packages
└── README.md
```

## Workflow

1. Load and inspect the data
2. Exploratory data analysis (matches per venue, top batsmen, top bowlers)
3. Label encoding of team, venue, batsman and bowler names
4. Feature selection using a correlation heatmap
5. 70/30 train-test split and Min-Max scaling (fit on training data only)
6. Build and train the neural network
7. Evaluate with MAE, RMSE and R2
8. Predict scores with `predict_score()` or the interactive widget

## Model

| Layer | Units | Activation |
|---|---|---|
| Input | 9 features | - |
| Dense | 512 | ReLU |
| Dense | 216 | ReLU |
| Output | 1 | Linear |

- Optimizer: Adam, Loss: Huber (delta 1.0)
- Epochs: 10, Batch size: 64
- Trainable parameters: 116,145

## Results

Evaluated on 22,805 held-out rows (30% test split):

| Metric | Value |
|---|---|
| MAE | 14.66 runs |
| RMSE | 19.99 runs |
| R2 | 0.526 |

## Sample Prediction

```python
predict_score(
    venue='M Chinnaswamy Stadium',
    bat_team='Chennai Super Kings',
    bowl_team='Royal Challengers Bangalore',
    batsman='MS Dhoni',
    bowler='P Kumar',
    runs=80, wickets=2, overs=10.0
)
# Predicted total: 166
```

Names must match the dataset exactly. The notebook also includes an `ipywidgets` form with dropdowns for the same inputs.

## Installation and Usage

Requires Python 3.10 to 3.12.

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `IPL_Score_Prediction.ipynb`, keep `ipl_data.csv` in the same folder, then choose Run All Cells. The notebook also runs on Google Colab (upload both files).

If the widget shows as text instead of dropdowns, run:

```bash
pip install --upgrade ipywidgets jupyterlab_widgets widgetsnbextension
```

then restart Jupyter.

## Limitations

- Data covers only the 2008-2017 seasons.
- Rows are split randomly, so balls from the same match appear in both train and test sets, which makes the test score optimistic.
- Label encoding gives names an artificial numeric order.
- Trained for only 10 epochs with no hyperparameter tuning.

## Future Work

- Split by match id so the test set contains unseen matches
- Use embedding or one-hot layers for teams, venues and players
- Add `runs_last_5` and `wickets_last_5` back as features
- Tune layers and epochs; add dropout and early stopping
- Train on newer seasons and deploy as a web app

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow / Keras, ipywidgets, Jupyter Notebook
