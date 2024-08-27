# Airline Passengers Prediction Using GRU
This project demonstrates the use of a Gated Recurrent Unit (GRU) model to predict the number of airline passengers over time using a time series dataset. The dataset contains monthly totals of international airline passengers from 1949 to 1960.

## Project Overview
Time series forecasting is a key application in various domains, including finance, weather prediction, and resource management. In this project, we utilize GRU, a type of recurrent neural network (RNN) designed to handle sequential data, to forecast future values in the time series.

### 1. Data Preprocessing
The dataset is loaded from a CSV file.
Data normalization is performed using MinMaxScaler to scale values between [0, 1], improving the model's performance.
The dataset is split into training (80%) and testing (20%) sets.
Data is restructured into a supervised learning format, using previous time_step values to predict the next value.

### 2. GRU Model
Architecture:
Two GRU layers with 50 units each.
A Dense layer with a single output neuron.
The model is compiled using the Adam optimizer and mean squared error (MSE) as the loss function.
The model is trained over 20 epochs with a batch size of 1.

### 3. Predictions
Predictions are generated for both the training and testing datasets.
The predicted values are inverse transformed to their original scale for comparison with the actual values.

### 4. Evaluation Metrics
Mean Absolute Error (MAE): Measures the average magnitude of errors in the predictions.
Mean Squared Error (MSE): Represents the average of the squared differences between the predicted and actual values.
Root Mean Squared Error (RMSE): The square root of MSE, providing an error metric in the same units as the original data.
The performance metrics for this model are:

Train MAE: 20.30
Test MAE: 34.41
Train MSE: 566.35
Test MSE: 1534.99
Train RMSE: 23.80
Test RMSE: 39.18

### 5. Visualization
A plot is generated to visualize the actual data alongside the model's predictions for both the training and testing sets, offering a clear comparison of model performance.

### 6. Model Saving
The trained GRU model is saved as air_passenger_gru_model.h5 for future use, allowing predictions to be made without retraining the model.

### Dependencies
Python 3.x
TensorFlow
scikit-learn
Matplotlib
NumPy
Pandas

### Running the Project

Clone the repository:
git clone https://github.com/yourusername/airline-passenger-prediction.git
cd airline-passenger-prediction

Install the required dependencies:
pip install -r requirements.txt

Run the Python script:
python gru_airline_passengers.py
The script will train the model, generate predictions, display a plot of the results, and print evaluation metrics to the console.

### Conclusion
This project illustrates the effectiveness of GRU networks for time series forecasting. Despite the complexities of sequential data, the GRU model successfully captures patterns in the airline passengers dataset, providing accurate predictions for both training and testing datasets.
