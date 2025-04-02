import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Input,LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix

data = pd.read_csv('/content/final_VAfeatures_36users.csv')
# print(data)


#CLUSTERING INTO 4 CLUSTERS

# Define clusters as empty sets to ensure uniqueness
user_assignments = {1: set(), 2: set(), 3: set(), 4: set()}

# Iterate through each row of the data to assign to clusters
for _, row in data.iterrows():
    user_id = int(row['P_id'])
    age = row['Age']
    gender = row['Gender']

    # Determine cluster assignment based on gender and age
    if gender == 1:
        if age in [1, 2]:
            user_assignments[1].add(user_id)  # Cluster 1: Gender 1 and Age 1 and 2
        elif age in [3, 4]:
            user_assignments[2].add(user_id)  # Cluster 2: Gender 1 and Age 3 and 4
    elif gender == 2:
        if age in [1, 2]:
            user_assignments[3].add(user_id)  # Cluster 3: Gender 2 and Age 1 and 2
        elif age in [3, 4]:
            user_assignments[4].add(user_id)  # Cluster 4: Gender 2 and Age 3 and 4

# Print results
for cluster_num, users in user_assignments.items():
    print(f"Cluster {cluster_num} contains users: {list(users)}")

print(user_assignments)

user_assignments = {
    1: {32, 1, 33, 34, 35, 6, 12, 16, 19, 21, 22, 23, 24, 26, 27, 29},
    2: {17, 18},
    3: {2, 3, 36, 4, 5, 8, 11, 13, 14, 15, 25, 28, 30, 31},
    4: {9, 10, 20, 7}
}

# Convert to the second structure
user_assignments_1 = {}
for cluster, users in user_assignments.items():
    for user in users:
        user_assignments_1[user] = cluster

print(user_assignments_1)


# ACTIVE LEARNING MODEL BASED ON LSTM


# ACTIVE LEARNING LSTM MODEL

# THRESHOLD = 0.3

#SPLIT FIRST AND THEN CREATE SEQUENCES USING SPLITTED SETS-TRAINING,RETRAINING AND TEST SET


# Initialize arrays to store TPR, FPR values, and probe counts
TPR_list = []
FPR_list = []
probe_count_list = []

# Function to split test data into initial 60% for retraining and final 40% for testing
def split_test_data(testX, testY, ratio=0.6):
    # Determine the index to split the data
    split_index = int(len(testX) * ratio)
    # First 60% for retraining
    retrainX, retrainY = testX[:split_index], testY[:split_index]
    # Last 40% for final testing
    final_testX, final_testY = testX[split_index:], testY[split_index:]

    return retrainX, final_testX, retrainY, final_testY

# Function to create dataset for LSTM (analogous to your XGBoost training/testing split)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back), 0:5])  # Input features
        dataY.append(dataset[(i + look_back) - 1, 5])  # Target labels
    return np.array(dataX), np.array(dataY)

feature_columns = ['Score', 'GSR_diff', 'HR_diff', 'valence_acc_video', 'arousal_acc_video']
look_back = 20  # Look-back window for LSTM
scaler = StandardScaler()

def get_training_and_testing_data(user, user_assignment, data):
    cluster = user_assignment.get(user)
    cluster_users = [u for u, c in user_assignment.items() if c == cluster and u != user]
    training_data = data[data['P_id'].isin(cluster_users)]
    testing_data = data[data['P_id'] == user]
    return training_data, testing_data

# Iterate over each user from 1 to 36
for user in range(1, 37):
    # Get training and testing data for the current user
    training_data, testing_data = get_training_and_testing_data(user, user_assignments_1, data)

    # Prepare training and testing data
    training_set = training_data[feature_columns].values
    testing_set = testing_data[feature_columns].values

    # Add the target column for labels (Probe)
    training_labels = training_data['Probe'].values
    testing_labels = testing_data['Probe'].values

    # Scale the training and testing set
    scaled_training_set = scaler.fit_transform(training_set)
    scaled_testing_set = scaler.transform(testing_set)

    # Create datasets for LSTM
    trainX, trainY = create_dataset(np.hstack([scaled_training_set, training_labels.reshape(-1, 1)]), look_back)

    # Split the test data into retraining (60%) and testing (40%) before creating dataset
    testX, final_testX, testY, final_testY = split_test_data(
        np.hstack([scaled_testing_set, testing_labels.reshape(-1, 1)]),
        testing_labels,
        ratio=0.6
    )

    # Create datasets for LSTM after splitting
    retrainX, retrainY = create_dataset(testX, look_back)
    final_testX, final_testY = create_dataset(final_testX, look_back)

    # Reshape the data for LSTM model: [samples, timesteps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
    retrainX = np.reshape(retrainX, (retrainX.shape[0], look_back, retrainX.shape[2]))
    final_testX = np.reshape(final_testX, (final_testX.shape[0], look_back, final_testX.shape[2]))

    # Build and train the LSTM model
    model = Sequential()
    model.add(Input(shape=(look_back, trainX.shape[2])))
    model.add(LSTM(10))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(trainX, trainY, epochs=35, batch_size=16, verbose=0)

    # Variable to store probe count for the current user
    probe_count = 0

    # Retraining phase using the retrainX set
    for i in range(len(retrainX)):
        # Predict the class label for the i-th sample in retrainX
        y_prob = model.predict(retrainX[i:i+1])[0][0]

        # If probability of class 1 is greater than 0.7, increase probe count and retrain the model
        if y_prob > 0.7:
            probe_count += 1
            # Concatenate the new row into the training set
            trainX = np.concatenate([trainX, retrainX[i:i+1]], axis=0)
            trainY = np.concatenate([trainY, retrainY[i:i+1]], axis=0)

            # Retrain the model
            model.fit(trainX, trainY, epochs=5, batch_size=16, verbose=0)

    # Final testing phase on the remaining 40% test data
    y_probs = (model.predict(final_testX) > 0.3).astype(int).flatten()
    final_testY = final_testY.flatten()

    # Calculate True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((y_probs == 1) & (final_testY == 1))
    FP = np.sum((y_probs == 1) & (final_testY == 0))
    TN = np.sum((y_probs == 0) & (final_testY == 0))
    FN = np.sum((y_probs == 0) & (final_testY == 1))

    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    # Append the TPR, FPR values, and probe count to the respective lists
    TPR_list.append(TPR)
    FPR_list.append(FPR)
    probe_count_list.append(probe_count)

    # Print the results for the current user
    print(f"User {user}:")
    print(f"  True Positive Rate (TPR): {TPR:.4f}")
    print(f"  False Positive Rate (FPR): {FPR:.4f}")
    print(f"  Probe Count: {probe_count}")

# Convert the lists to numpy arrays if needed
TPR_array = np.array(TPR_list)
FPR_array = np.array(FPR_list)
probe_count_array = np.array(probe_count_list)

# Calculate the average TPR, FPR, and probe count
average_TPR = np.mean(TPR_array)
average_FPR = np.mean(FPR_array)
average_probe_count = np.mean(probe_count_array)

# Print the final TPR, FPR arrays, and probe counts
print("Final TPR values for all users:", TPR_array)
print("Final FPR values for all users:", FPR_array)
print("Final probe counts for all users:", probe_count_array)

# Print the average TPR, FPR, and probe count
print(f"\nAverage True Positive Rate (TPR): {average_TPR:.4f}")
print(f"Average False Positive Rate (FPR): {average_FPR:.4f}")
print(f"Average Probe Count: {average_probe_count:.4f}")

# #VIDEO ID WISE LSTM MODEL

# # Initialize arrays to store TPR, FPR values, and probe counts
# TPR_list = []
# FPR_list = []
# probe_count_list = []
# probe_counts_per_video = {}  # To store probe counts for each video_id

# # Function to split test data into initial 60% for retraining and final 40% for testing
# def split_test_data(testX, testY, ratio=0.6):
#     split_index = int(len(testX) * ratio)
#     retrainX, retrainY = testX[:split_index], testY[:split_index]
#     final_testX, final_testY = testX[split_index:], testY[split_index:]
#     return retrainX, final_testX, retrainY, final_testY

# # Function to create dataset for LSTM (analogous to your XGBoost training/testing split)
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         dataX.append(dataset[i:(i + look_back), 0:5])  # Input features
#         dataY.append(dataset[(i + look_back) - 1, 5])  # Target labels
#     return np.array(dataX), np.array(dataY)

# feature_columns = ['Score', 'GSR_diff', 'HR_diff', 'valence_acc_video', 'arousal_acc_video']
# look_back = 20  # Look-back window for LSTM
# scaler = StandardScaler()

# def get_training_and_testing_data(user, user_assignment, data, video_id):
#     cluster = user_assignment.get(user)
#     cluster_users = [u for u, c in user_assignment.items() if c == cluster and u != user]
#     training_data = data[(data['P_id'].isin(cluster_users)) & (data['video_id'] == video_id)]
#     testing_data = data[data['P_id'] == user]
#     return training_data, testing_data

# # Iterate over each user from 1 to 36
# for user in range(1, 37):
#     probe_counts_per_video[user] = {}

#     # Train the model on data of similar users (in the same cluster) for video_id = 1
#     training_data, _ = get_training_and_testing_data(user, user_assignments_1, data, video_id=1)
#     training_set = training_data[feature_columns].values
#     training_labels = training_data['Probe'].values
#     scaled_training_set = scaler.fit_transform(training_set)

#     # Create dataset for LSTM with look_back window
#     trainX, trainY = create_dataset(np.hstack([scaled_training_set, training_labels.reshape(-1, 1)]), look_back)
#     trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))

#     # Build and train the LSTM model
#     model = Sequential()
#     model.add(Input(shape=(look_back, trainX.shape[2])))
#     model.add(LSTM(10))
#     model.add(Dropout(0.5))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     # Train the model with video_id = 1 of similar users
#     model.fit(trainX, trainY, epochs=35, batch_size=16, verbose=0)

#     # Probe count for video_id = 1 of user 1
#     _, testing_data = get_training_and_testing_data(user, user_assignments_1, data, video_id=1)
#     testing_set = testing_data[feature_columns].values
#     testing_labels = testing_data['Probe'].values
#     scaled_testing_set = scaler.transform(testing_set)

#     # Create dataset for user's video_id = 1 and calculate probe count
#     userX, userY = create_dataset(np.hstack([scaled_testing_set, testing_labels.reshape(-1, 1)]), look_back)
#     userX = np.reshape(userX, (userX.shape[0], look_back, userX.shape[2]))

#     probe_count = 0
#     for i in range(len(userX)):
#         y_prob = model.predict(userX[i:i+1])[0][0]
#         if y_prob > 0.7:
#             probe_count += 1
#     probe_counts_per_video[user][1] = probe_count  # Store probe count for video_id = 1

#     # Now, retrain the model with video_id = 2 to 6 of user 1 and keep track of probe count for each video_id
#     for video_id in range(2, 7):
#         _, retrain_data = get_training_and_testing_data(user, user_assignments_1, data, video_id=video_id)
#         retrain_set = retrain_data[feature_columns].values
#         retrain_labels = retrain_data['Probe'].values
#         scaled_retrain_set = scaler.transform(retrain_set)

#         retrainX, retrainY = create_dataset(np.hstack([scaled_retrain_set, retrain_labels.reshape(-1, 1)]), look_back)
#         retrainX = np.reshape(retrainX, (retrainX.shape[0], look_back, retrainX.shape[2]))

#         probe_count_video = 0
#         for i in range(len(retrainX)):
#             y_prob = model.predict(retrainX[i:i+1])[0][0]
#             if y_prob > 0.7:
#                 probe_count_video += 1
#                 trainX = np.concatenate([trainX, retrainX[i:i+1]], axis=0)
#                 trainY = np.concatenate([trainY, retrainY[i:i+1]], axis=0)
#                 model.fit(trainX, trainY, epochs=5, batch_size=16, verbose=0)
#         probe_counts_per_video[user][video_id] = probe_count_video

#     # Test the model on video_id 7 and 8 of user 1
#     _, test_data = get_training_and_testing_data(user, user_assignments_1, data, video_id=7)
#     test_set = test_data[feature_columns].values
#     test_labels = test_data['Probe'].values
#     scaled_test_set = scaler.transform(test_set)

#     final_testX, final_testY = create_dataset(np.hstack([scaled_test_set, test_labels.reshape(-1, 1)]), look_back)
#     final_testX = np.reshape(final_testX, (final_testX.shape[0], look_back, final_testX.shape[2]))

#     y_probs = (model.predict(final_testX) > 0.3).astype(int).flatten()
#     final_testY = final_testY.flatten()

#     TP = np.sum((y_probs == 1) & (final_testY == 1))
#     FP = np.sum((y_probs == 1) & (final_testY == 0))
#     TN = np.sum((y_probs == 0) & (final_testY == 0))
#     FN = np.sum((y_probs == 0) & (final_testY == 1))

#     TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
#     FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

#     TPR_list.append(TPR)
#     FPR_list.append(FPR)

#     print(f"User {user}:")
#     print(f"  True Positive Rate (TPR): {TPR:.4f}")
#     print(f"  False Positive Rate (FPR): {FPR:.4f}")
#     print(f"  Probe Count for each video_id: {probe_counts_per_video[user]}")

# # Convert the lists to numpy arrays if needed
# TPR_array = np.array(TPR_list)
# FPR_array = np.array(FPR_list)

# # Calculate the average TPR and FPR
# average_TPR = np.mean(TPR_array)
# average_FPR = np.mean(FPR_array)

# print("Final TPR values for all users:", TPR_array)
# print("Final FPR values for all users:", FPR_array)

# print(f"\nAverage True Positive Rate (TPR): {average_TPR:.4f}")
# print(f"Average False Positive Rate (FPR): {average_FPR:.4f}")

