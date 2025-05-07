# New York City Taxi Fare Prediction

<h1> 📌 Problem Statement:</h1>
<b>
The New York City Taxi Fare Prediction challenge is a regression problem where the goal is to predict the fare amount (in USD) that a passenger will have to pay for a taxi ride in New York City, based on certain information available at the start of the ride.
</b>

<h2>
 📂 Dataset that includes:
</h2>

- Pickup date and time

- Pickup location (latitude and longitude)

- Drop-off location (latitude and longitude)

- Passenger count


The task is to use this information to accurately predict the taxi fare for each trip.

<h2>
 Key Features Typically Available in the Dataset:
</h2>

- pickup_datetime: When the taxi ride started (timestamp).

- pickup_longitude, pickup_latitude: The pickup location's GPS coordinates.

- dropoff_longitude, dropoff_latitude: The drop-off location's GPS coordinates.

- passenger_count: Number of passengers.

- fare_amount: The target variable (how much the trip cost).

We'll train a machine learning model to predict the fare for a taxi ride in New York city given information like pickup date & time, pickup location, drop location and no. of passengers.

Dataset Link: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

<h2>
 Challenges in the Problem:
</h2>

 
- Outliers: Extremely high or negative fare amounts.

- Noise: Incorrect GPS coordinates (e.g., latitudes > 90 or longitudes > 180).

- Feature Engineering: Need to derive meaningful features, such as:

    - Distance traveled (haversine distance).

    - Time-based features (day of week, hour of day).

    - Traffic patterns, holidays, weather (if available).


<h2>
 Typical Steps in a Solution:
</h2>

- Download the Dataset

   - Install required libraries
   - Download data from Kaggle
   - View dataset files
   - Load training set with Pandas
   - Load test set with Pandas




- Loading Training Set

    Loading the entire dataset into Pandas is going to be slow, so we can use the following optimizations:

     - Ignore the `key` column
     - Parse pickup datetime while loading data
     - Specify data types for other columns
     - `float32` for geo coordinates
     - `float32` for fare amount
     - `uint8` for passenger count
     - Work with a 1% sample of the data (~500k rows)

- 📊 Exploratory Data Analysis and Visualization

 - Prepare Dataset for Training

   - Split Training & Validation Set
   - Fill/Remove Missing Values
   - Extract Inputs & Outputs
     - Training
     - Validation
     - Test


- Data Cleaning: Remove or correct invalid coordinates, negative fares, unrealistic passenger counts, etc.

- Feature Engineering:

   - Add distance between pickup & drop

   - Extract parts of date (weekday, hour, ywar , month , days).



- Train & Evaluate Different Models

    We'll train each of the following & submit predictions to Kaggle:

    - Ridge Regression
    - Random Forests
    - Gradient Boosting
    - Decision Tree

- Evaluation: Metric usually used is Root Mean Squared Error (RMSE).

- Tune Hyperparmeters

    We'll train parameters for the XGBoost model. Here’s a strategy for tuning hyperparameters:

   - Tune the most important/impactful hyperparameter first e.g. n_estimators , max_depth m learning_rate

   - With the best value of the first hyperparameter, tune the next most impactful hyperparameter

<h2>
 Project OverView:
</h2>

 **1. Install Dependencies**
```

      !pip install numpy pandas matplotlib seaborn opendatasets scikit-learn xgboost --quiet
```


**2. Explore the Dataset**

   Observations about training data:

   - 550k+ rows, as expected
   - No missing data (in the sample)
   - `fare_amount` ranges from \$-52.0 to \$499.0
   - `passenger_count` ranges from 0 to 208
   - There seem to be some errors in the latitude & longitude values
   - Dates range from 1st Jan 2009 to 30th June 2015
   - The dataset takes up ~19 MB of space in the RAM

   We may need to deal with outliers and data entry errors before we train our model.


   Some observations about the test set:

   - 9914 rows of data
   - No missing values
   - No obvious data entry errors
   - 1 to 6 passengers (we can limit training data to this range)
   - Latitudes lie between 40 and 42
   - Longitudes lie between -75 and -72
   - Pickup dates range from Jan 1st 2009 to Jun  30th 2015 (same as training set)

   We can use the ranges of the test set to drop outliers/invalid data from the training set.


**3. Exploratory Data Analysis and Visualization**

   - Q What is the busiest day of the week?

       We analyzed the pickup_datetime_weekday column. The counts for each day (where 0 = Monday and 6 = Sunday) are as follows:

        - Friday (4): 66,613 rides
        
        - Saturday (5): 65,530 rides
        
        - Thursday (3): 64,887 rides
        
        - Wednesday (2): 62,539 rides
        
        - Tuesday (1): 60,576 rides
        
        - Sunday (6): 56,669 rides
        
        - Monday (0): 55,649 rides

     Friday is the busiest day for taxi pickups, followed closely by Saturday and Thursday. This suggests that taxi demand increases towards the end of the workweek


  - Q In which month are fares the highest?

      Here are the Total fare amounts per month (where 1 = January, 12 = December):

      - June (6): $474,170.25

      - May (5): $452,513.34
      
      - April (4): $447,959.50
      
      - July (7): $445,795.69
      
      - February (2): $412,085.47
      
      - November (11): $405,111.59
      
      - March (3): $393,834.31
      
      - December (12): $389,207.75
      
      - October (10): $388,029.69
      
      - September (9): $377,153.19
      
      - August (8): $364,826.38
      
      - January (1): $351,812.97
   
    The month with the highest average fare is June, followed closely by May and April. This trend may be influenced by increased travel, tourism, and warmer weather activities during late spring and early summer.



**4 Split Training & Validation Set**

  We'll set aside 20% of the training data as the validation set, to evaluate the models we train on previously unseen data.
  Since the test set and training set have the same date ranges, we can pick a random 20% fraction.

  ```
   len(train_df) , len(val_df)
  ```
  Training data has a 443528 rows and 
  validation data has a  110882 rows


**5 Train Baseline Models**

   - Train & Evaluate Hardcoded Model
  
   - Let's create a simple model that always predicts the average.

 
  ```
   import numpy as np
class MeanRegressor():
  def fit(self , inputs , targets):
    self.mean = targets.mean()

  def predict(self , inputs):
    return np.full(inputs.shape[0] , self.mean)
  ```

  ```
mean_model = MeanRegressor()
  ```

  ```
mean_model.fit(train_inputs , train_target)
  ```

  ```
train_preds = mean_model.predict(train_inputs)
train_preds
  ```

  ```
val_preds = mean_model.predict(val_inputs)
val_preds
  ```

The evaluation metric for this competition is the root mean-squared error or RMSE. RMSE measures the difference between the predictions of a model, and the corresponding ground truth. A large RMSE is equivalent to a large average error, so smaller values of RMSE are better.

RMSE is given by:

RMSE=1n∑i=√1n(y^i−yi)2

  ```
from sklearn.metrics import mean_squared_error , root_mean_squared_error
train_rmse = root_mean_squared_error(train_target, train_preds)
  ```

  ```
train_rmse
  ```

  ```
val_rmse = root_mean_squared_error(val_target , val_preds )
val_rmse
  ```

**6.Train & Evaluate Baseline Model**

We'll traina linear regression model as our baseline, which tries to express the target as a weighted sum of the inputs.

  ```
from sklearn.linear_model import LinearRegression
  ```
  ```
linreg_model = LinearRegression()
  ```

  ```
linreg_model.fit(train_inputs , train_target)
  ```

  ```
train_preds = linreg_model.predict(train_inputs)
train_preds
  ```

  ```
val_preds = linreg_model.predict(val_inputs)
val_preds
  ```

  ```
train_rmse = root_mean_squared_error(train_target , train_preds)
train_rmse
  ```
  ```
val_rmse = root_mean_squared_error(val_target , val_preds)
val_rmse
  ```

The linear regression model is off by $9.79, which isn't much better than simply predicting the average.

This is mainly because the training data  is not in a format that's useful for the model, and we're not using one of the most important columns: pickup date & time.

However, now we have a baseline that our other models should ideally beat.
   


**7 Feature Engineering**

  **Add Distance Between Pickup and Drop**

  We can use the haversine distance:
 <a href = "https://en.wikipedia.org/wiki/Haversine_formula"> haversine distance  </a>

  ```
def haversine_np(lon1 , lat1 , lon2 , lat2):

  lon1 , lat1 , lon2 , lat2 = map(np.radians , [lon1 , lat1 , lon2 , lat2])

  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

  c = 2 * np.arcsin(np.sqrt(a))
  km = 6367 * c
  return km
  ```
 
**7 Train & Evaluate Different Models**

   - Ridge Regression

      Ridge Regression, a regularized linear regression model that helps reduce overfitting by penalizing large coefficients. It’s especially useful when dealing with multicollinearity in the features.

     After training the model, we evaluated its performance using Root Mean Squared Error (RMSE) on both the training and validation sets.

     - Training RMSE: 5.10927939051802
    
     - Validation RMSE: 5.060686737738835
    
     The training and validation RMSE scores are quite close, suggesting that the Ridge Regression model is not overfitting and generalizes well to unseen data.
     
   - Random Forests

      A Random Forest Regressor — an ensemble method that builds multiple Decision Trees and averages their predictions.

     We then evaluated its performance using Root Mean Squared Error (RMSE):

     - Training RMSE: 3.5764
    
     - Validation RMSE: 3.9976
    
    The Random Forest model achieved a significantly lower RMSE than Ridge Regression . The gap between training and validation RMSE is small, indicating good generalization with Minmum overfitting.
    
       
   - Gradient Boosting

      To push the model performance further, we used XGBoost (Extreme Gradient Boosting)

     Performance evaluation using Root Mean Squared Error (RMSE):

      - Training RMSE: 3.1619
    
      - Validation RMSE: 3.9022

      XGBoost achieved the lowest validation RMSE among all models tested, indicating it was the most effective in capturing complex patterns in the data. Although the training RMSE is lower than the validation RMSE , the gap remains reasonable — showing that the model generalizes well and is not  overfitting.
        




   - Decision Tree

     we trained a Decision Tree Regressor, which can model non-linear relationships by splitting data based on feature thresholds.

     Performance:

     - Training RMSE: 5.1172
     
     - Validation RMSE: 5.2848
    
   The Decision Tree showed a slightly higher validation RMSE , indicating limited performance at the chosen max_depth



| Model            | Training RMSE | Validation RMSE |
| ---------------- | ------------- | --------------- |
| Ridge Regression | 5.1092        | 5.0607          |
| Decision Tree    | 5.1172        | 5.2848          |
| Random Forest    | 3.5764        | 3.9976          |
| XGBoost          | 3.1619        | 3.9022          |

**Best Model: XGBoost continues to outperform others in terms of validation RMSE.**





