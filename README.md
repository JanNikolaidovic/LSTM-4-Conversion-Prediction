

# Customer Journey Prediction Model Explanation

At its core, this code predicts the likelihood of a sale based on a customer's journey, represented as a sequence of interactions or steps using an LSTM-based deep learning model.

## Libraries and Data Loading
- **Libraries**: We use `numpy` and `pandas` for data wrangling, `tensorflow` and `keras` for deep learning, and `scikit-learn` for other tasks.
- **Data**: The data, sourced from `dt_01.csv`, is not available in the repo to ensure confidentiality. Each row in the DataFrame `df` represents an interaction step, with the 'sale' column indicating a sale occurrence.

## Data Preprocessing
- Categorical columns like 'channel', 'season', and 'period' are one-hot encoded.
- Sequences are standardized in length using `pad_sequences`, typically to a length of 20.

## Data Splitting
The dataset is partitioned as:
- **Training**: 60%
- **Validation**: 20%
- **Testing**: 20%

## Model Architecture
The LSTM-based model includes:
- A masking layer
- An LSTM layer with 64 units
- A dropout layer for regularization
- A dense output layer for binary classification

## Model Training
- **Optimizer**: Adam
- **Loss Function**: Binary cross-entropy
- **Early Stopping**: If validation loss doesn't improve for five epochs, training halts to prevent overfitting.

## Evaluation
- We chart the model's accuracy and loss for the training and validation sets.
- A confusion matrix contrasts the model's predictions with the real test data outcomes.

## Counterfactual Analysis
- By altering sequences initially marked as non-conversions, we gauge the effect of different starting interactions on the model's decisions, offering insights into potential alternate customer journeys.

