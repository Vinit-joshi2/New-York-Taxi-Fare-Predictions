# New York City Taxi Fare Prediction

<h1>problem Statement:</h1>
<b>
The New York City Taxi Fare Prediction challenge is a regression problem where the goal is to predict the fare amount (in USD) that a passenger will have to pay for a taxi ride in New York City, based on certain information available at the start of the ride.
</b>

<h2>
 Dataset that includes:
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





