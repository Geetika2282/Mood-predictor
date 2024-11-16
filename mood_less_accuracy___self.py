


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:\\Users\\geet1\\ML\\Mood predictor\\Mood-based_recommendation\\data\\Daylio_Abid.csv')



"""## Step 1: Data Cleaning and Preprocessing

"""

df['full_date'] = pd.to_datetime(df['full_date'])
# print(df.isnull().sum())

# replacing null values with most frequent values of 'activities' column
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
si_obj = SimpleImputer(strategy='most_frequent')
s1 = pd.Series(df['activities'])
s1 = pd.Series(si_obj.fit_transform(df[['activities']]).ravel())
df['activities'] = s1

# convert time to category
df['time'] = pd.to_datetime(df['time'], format='%I:%M %p').dt.strftime('%H:%M')
df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
bins = [0, 5, 12, 17, 24]  # Time ranges: [0-5) morning, [5-12) afternoon, [12-17) evening, [17-24) night
labels = ['night', 'morning', 'afternoon', 'evening']

# Classify time of day using pd.cut
df['time_of_day'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)

# df.drop(columns=['time', 'hour'], inplace=True)
# print(df.columns)

# Drop unnecessary columns
df.drop(columns=['full_date', 'date', 'hour', 'sub_mood'], inplace=True)

# Display the updated DataFrame
# print(df)



# split the 'activities' column
df['activities'] = df['activities'].str.split('|')

from sklearn.preprocessing import MultiLabelBinarizer


# Use MultiLabelBinarizer to One-Hot Encode the activities
mlb = MultiLabelBinarizer()
activities_encoded = pd.DataFrame(mlb.fit_transform(df['activities']), columns=mlb.classes_)

# Concatenate the encoded activities back to the dataframe
df_encoded = pd.concat([df.drop(columns=['activities']), activities_encoded], axis=1)

# Handle missing data by imputing
imputer = SimpleImputer(strategy='most_frequent')
# Use MultiLabelBinarizer to One-Hot Encode the activities
mlb = MultiLabelBinarizer()
activities_encoded = pd.DataFrame(mlb.fit_transform(df['activities']), columns=mlb.classes_)

# Concatenate the original DataFrame with the new activity columns
df_encoded = pd.concat([df.drop(columns=['activities']), activities_encoded], axis=1)

# Display the new DataFrame with individual activity columns
# print(df_encoded)

activity_col = (df_encoded.iloc[:, 4:])
activity_col.columns.tolist()

df_encoded['mood'].unique()

# creating a dict based on mood suggesting activities

mood_activity_dict = {
    'Good': ['reading', 'prayer', 'walk', 'exercise', 'yoga', 'meditation', 'coding', 'travel', 'good meal', 'family', 'learning'],
    'Normal': ['reading', 'learning', 'prayer', 'watching series', 'news update', 'cleaning', 'family', 'coding', 'email', 'shopping'],
    'Awful': ['sleeping', 'resting', 'taking a break', 'watching movies', 'listening to music', 'podcast', 'power nap'],
    'Amazing': ['traveling', 'meeting friends', 'exploring', 'adventure', 'yoga', 'partying', 'new things', 'outdoor activities'],
    'Bad': ['resting', 'watching TV', 'sleeping', 'taking a break', 'meditation', 'journaling', 'shower', 'quiet time', 'research']
}

# Select columns up to the 5th column (index 0-4) and drop the rest
df_encoded = df_encoded.iloc[:, :4]
# print(df_encoded)



"""## Questions
1. "How much sleep did you get last night?"
2. "How stressful was your day on a scale of 1 to 10?"
3. "Did you have any physical activity today?"
4. "How social were you today? (Scale of 1 to 10)"
5. "How much did you enjoy your activities today?"
6. "Did you face any challenges or difficulties today?"
7. "On a scale of 1 to 10, how motivated did you feel today?"

Adding random data, new columns required for prediction
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate random values for each column
hours_of_sleep = np.random.randint(4, 9, size=940)  # Random hours between 4 and 8
stress_level = np.random.randint(1, 11, size=940)   # Random stress level between 1 and 10
physical_activity = np.random.choice([0, 1], size=940)  # 0 for No, 1 for Yes
social_interaction_level = np.random.randint(1, 11, size=940)  # Random social interaction level between 1 and 10
activity_enjoyment_level = np.random.randint(1, 6, size=940)  # Random enjoyment level between 1 and 5
faced_challenges = np.random.choice([0, 1], size=940)  # 0 for No, 1 for Yes
motivation_level = np.random.randint(1, 11, size=940)  # Random motivation level between 1 and 10

# Create a DataFrame to hold these random values
random_data = pd.DataFrame({
    'hours_of_sleep': hours_of_sleep,
    'stress_level': stress_level,
    'physical_activity': physical_activity,
    'social_interaction_level': social_interaction_level,
    'activity_enjoyment_level': activity_enjoyment_level,
    'faced_challenges': faced_challenges,
    'motivation_level': motivation_level
})

random_data.head()

"""Adding these columns to actual dataframe"""

# Assuming 'df' is your existing dataframe and 'random_data' is the new dataframe with random values
df_encoded = pd.concat([df, random_data], axis=1)
df_encoded.drop(columns=['time'], inplace=True)
# Check the updated dataframe
df_encoded.head()



# print(df_encoded.columns)

# print(df_encoded.info())

"""### Mood encoded"""

from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()

# Label encode 'mood' column
df_encoded['mood'] = le.fit_transform(df_encoded['mood'])

# Save the mapping for future use
mood_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# print(mood_mapping)

# print(df_encoded)

"""### Label weekday"""

from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()

# Label encode 'weekday' and 'mood' columns
df_encoded['weekday'] = le.fit_transform(df_encoded['weekday'])
df_encoded['mood'] = le.fit_transform(df_encoded['mood'])

# Check the encoding
# print(df_encoded[['weekday', 'mood']].head())

# print(df_encoded)

"""### Map time_of_day"""

from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le_time = LabelEncoder()

# Fit and transform the 'time_of_day' column
df_encoded['time_of_day'] = le_time.fit_transform(df_encoded['time_of_day'])

# Check the mapping
time_of_day_mapping = dict(zip(le_time.classes_, le_time.transform(le_time.classes_)))
# print(time_of_day_mapping)

# {'afternoon': 0, 'evening': 1, 'morning': 2, 'night': 3}

# print(df_encoded)



"""## Prepare the data"""

X = df_encoded.drop(columns=['mood', 'activities'], axis=1)
y = df_encoded['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Train the SVC"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train, y_train)
# {'sigmoid', 'linear', 'rbf', 'poly', 'precomputed'}

# predict and check accuracy

y_pred = svc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# print("Classification Report:")
# print(classification_report(y_test, y_pred))

"""Predicting by passing an array, getting mood (str) as output"""

# Example feature values for a single prediction
single_instance = [[7, 8, 4, 0, 1, 6, 4, 1, 8]]  # This should match the order of your features

# Predict mood for this single instance
y_pred = svc.predict(single_instance)

# Convert the predicted integer to the corresponding mood label
mood_mapping = {0: 'Good', 1: 'Normal', 2: 'Awful', 3: 'Amazing', 4: 'Bad'}
predicted_mood = mood_mapping[y_pred[0]]  # y_pred is an array, so we use [0] for the first prediction

print(predicted_mood)  # Prints the predicted mood

mood_to_activity_dict = {
    'Good': ['Exercise', 'Yoga', 'Travel', 'Walk', 'Cooking'],
    'Normal': ['Reading', 'Learning', 'Shopping', 'Watching Series'],
    'Awful': ['Meditation', 'Listening to Music', 'Power Nap'],
    'Amazing': ['Party', 'Travel', 'Socialize'],
    'Bad': ['Reading', 'Gaming', 'Solo Activities']
}

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Example mapping of moods to activities
mood_activity_dict = {
    'Good': ['exercise', 'yoga', 'reading', 'family time', 'music'],
    'Normal': ['work', 'learning', 'shopping', 'cooking', 'movies'],
    'Awful': ['resting', 'meditation', 'listening to music', 'writing'],
    'Amazing': ['party', 'travel', 'socializing', 'hiking', 'adventure sports'],
    'Bad': ['sleeping', 'writing', 'taking a walk', 'watching series', 'journaling']
}

# Mood categories (ensure the same mapping as used in your training data)
mood_labels = ['Good', 'Normal', 'Awful', 'Amazing', 'Bad']
mood_mapping = {i: mood_labels[i] for i in range(len(mood_labels))}

# Example: Making a prediction for a new user input
user_input = [7, 5, 1, 1, 7, 2, 4, 0, 9]  # Example input (hours_of_sleep, stress_level, physical_activity, etc.)

# Prepare your training data (just a quick example here)
# In practice, use your actual dataframe to get X_train and y_train
# Assuming you have already split the data
X_train = X  # Replace with your actual features
y_train = y  # Replace with your target variable (mood)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
svc = SVC(kernel='linear')
svc.fit(X_train_scaled, y_train)

# Standardize the user input and make prediction
user_input_scaled = scaler.transform([user_input])
predicted_mood_int = svc.predict(user_input_scaled)[0]

# Map predicted integer to mood label
predicted_mood = mood_mapping[predicted_mood_int]

# Recommend activities based on predicted mood
recommended_activities = mood_activity_dict.get(predicted_mood, [])

# Output results
print(f"Predicted Mood: {predicted_mood}")
print(f"Recommended Activities: {recommended_activities}")

"""# Saving the model using Joblib"""


# pip install openai==0.28.0
#
# import openai
#
# # Set your OpenAI API key
# openai.api_key = "your_api_key_here"
#
# def generate_image(prompt):
#     try:
#         response = openai.Image.create(
#             prompt=prompt,
#             n=1,  # Number of images to generate
#             size="512x512"  # Image size
#         )
#         # Get the image URL
#         image_url = response['data'][0]['url']
#         return image_url
#     except Exception as e:
#         print(f"Error generating image: {e}")
#         return None
#
# # Recommend activities with images
# def recommend_activities_with_images(mood):
#     activities = mood_activity_dict.get(mood, [])
#     activities_with_images = []
#
#     for activity in activities:
#         image_url = generate_image(f"{activity} illustration")
#         activities_with_images.append((activity, image_url))
#
#     return activities_with_images
#
# predicted_mood = "Good"  # Example mood
# recommendations = recommend_activities_with_images(predicted_mood)
#
# print(f"Recommended activities for mood '{predicted_mood}':")
# for activity, image_url in recommendations:
#     print(f"- {activity}: {image_url}")
#
