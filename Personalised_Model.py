import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
import pandas as pd
pd.set_option('display.max_rows', None)


data_set = pd.read_csv('/content/EMG-EPN.csv')
user_id = data_set['user_id']
df = pd.DataFrame(data_set)

# df['new_gestures'] = df['gesture'] - 1                 #make a new column new_gestures which has values ranging from 0-5 instead of 1-6



array = []                                             #array to store accuracies
array_f1 = []
overall_confusion_matrix = None

df['index'] = df.index + 1  # Adding 1 to start the index from 1


for user_id in range (1,307):

    # if user_id == 34:
    #     continue
                                  #loop starting from user_1 to user_36
    data_subset = df[df['user_id'] == user_id]  # Filter data for each user

    grouped_df = data_subset.groupby('gesture')

    train_dfs = []
    test_dfs = []
    for gesture, group in grouped_df:
      # Determine the split index
      train_index = int(0.80 * len(group))
      test_index = int(0.20 * len(group))

      # Split the data
      train_data = group.head(train_index)
      test_data = group.tail(test_index)
      print('len of gesture_i train_data =',len(train_data))
      print('len of gesture_i test_data =',len(test_data))

    # Append to respective lists
      train_dfs.append(train_data)
      test_dfs.append(test_data)

    # Concatenate all the training and testing data
    train_set = pd.concat(train_dfs)
    test_set = pd.concat(test_dfs)

    # print('Train set- ',train_set)
    # print('Test set- ',test_set)

    train_set = train_set.sort_values(by='index')
    Y_train = train_set['gesture']
    X_train = train_set.drop(['user_id','index','gestureName','gesture'],axis='columns')

    test_set = test_set.sort_values(by='index')
    Y_test = test_set['gesture']
    X_test = test_set.drop(['user_id','index','gestureName','gesture'],axis='columns')


    #test set should contain last 20% of the data
    print('len of training_set=',len(train_set))
    print('len of test_set=',len(test_set))


    # Counting the number of instances of gesture 0
    # num_gesture_0 = (test_set['gesture'] == 0).sum()
    # print('Number of instances of gesture 0 in test set =', num_gesture_0)

    # num_gesture_1 = (test_set['gesture'] == 1).sum()
    # print('Number of instances of gesture 1 in test set =', num_gesture_1)

    # num_gesture_2 = (test_set['gesture'] == 2).sum()
    # print('Number of instances of gesture 2 in test set =', num_gesture_2)

    # num_gesture_3 = (test_set['gesture'] == 3).sum()
    # print('Number of instances of gesture 3 in test set =', num_gesture_3)

    # num_gesture_4 = (test_set['gesture'] == 4).sum()
    # print('Number of instances of gesture 4 in test set =', num_gesture_4)

    # num_gesture_5 = (test_set['gesture'] == 5).sum()
    # print('Number of instances of gesture 5 in test set =', num_gesture_5)

    # num_gesture_6 = (test_set['gesture'] == 6).sum()
    # print('Number of instances of gesture 6 in test set =', num_gesture_6)

    # num_gesture_7 = (test_set['gesture'] == 7).sum()
    # print('Number of instances of gesture 7 in test set =', num_gesture_7)

    # num_gesture_8 = (test_set['gesture'] == 8).sum()
    # print('Number of instances of gesture 8 in test set =', num_gesture_8)

    # num_gesture_9 = (test_set['gesture'] == 9).sum()
    # print('Number of instances of gesture 9 in test set =', num_gesture_9)

    # num_gesture_10 = (test_set['gesture'] == 10).sum()
    # print('Number of instances of gesture 10 in test set =', num_gesture_10)






    model = xgb.XGBClassifier()                                             #training the model
    model.fit(X_train, Y_train)

    X_train_prediction = model.predict(X_train)

    X_test_prediction = model.predict(X_test)                                                 #evalution of model
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)                            #calculating accuracy values

    train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    test_data_accuracy = 100*test_data_accuracy
    array.append(test_data_accuracy)                                                           #storing the accuracy

    f1 = f1_score(X_test_prediction, Y_test, average = 'macro')
    array_f1.append(f1)

    print('User {user_id} accuracy = ',test_data_accuracy)

    user_confusion_matrix = confusion_matrix(Y_test,X_test_prediction )

    # Update overall confusion matrix
    if overall_confusion_matrix is None:
        overall_confusion_matrix = user_confusion_matrix
    else:
        overall_confusion_matrix += user_confusion_matrix

print(array)
print(array_f1)
print(len(array))
print(overall_confusion_matrix)
mean_accuracy = np.mean(array)
std_dev = np.std(array)
mean_f1 = np.mean(array_f1)

print('mean =',mean_accuracy)
print('std_dev =',std_dev)
print('mean_f1 =',mean_f1)

print(np.std(array_f1))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(overall_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

import matplotlib.pyplot as plt

# Assuming array contains the accuracy scores for all users except user 34
# Also assuming array_f1 contains the F1 scores for the same users

# User IDs renumbered from 1 to 35
user_ids_renumbered = list(range(1, len(array) + 1))

# Ensure the lengths match
assert len(user_ids_renumbered) == len(array), "Mismatch between user_ids and accuracy array lengths"
assert len(user_ids_renumbered) == len(array_f1), "Mismatch between user_ids and F1 score array lengths"

# Plot bar graph for accuracy scores
plt.figure(figsize=(14, 6))  # Adjust figure size if needed
bars = plt.bar(user_ids_renumbered, array, color='skyblue')  # Customize color if needed

# Add text labels for accuracy scores
for bar, score in zip(bars, array):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.0f}',
             ha='center', va='bottom', fontsize=6)

# Configure plot labels and appearance
plt.xlabel('User ID')
plt.ylabel('Accuracy Score')
plt.title('Bar Graph of Accuracy Scores by User ID (Personalized Model)')
plt.xticks(user_ids_renumbered, rotation=45, fontsize=6)  # Rotate x-axis labels

# Add grid and adjust layout
plt.grid(axis='y')
plt.tight_layout()

# Display the plot
plt.show()


user_ids = range(1, 307)  # User IDs from 1 to 36

# # Plot bar graph
# plt.figure(figsize=(14, 6))  # Adjust figure size if needed
# bars = plt.bar(user_ids, array, color='skyblue')  # Customize color if needed

# for bar, score in zip(bars, array):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.0f}', ha='center', va='bottom', fontsize = 6)

# plt.xlabel('User ID')
# plt.ylabel('Accuracy Score')
# plt.title('Bar Graph of Accuracy Scores by User ID(Personalised Model)')
# # plt.xticks(rotation=25)  # Rotate x-axis labels for better readability
# plt.xticks(user_ids,rotation = 45,fontsize = 6)

# plt.grid(axis='y')  # Show grid lines on the y-axis
# plt.tight_layout()  # Adjust layout to prevent clipping of labels
# plt.show()

print('sum of enteries =',np.sum(overall_confusion_matrix))