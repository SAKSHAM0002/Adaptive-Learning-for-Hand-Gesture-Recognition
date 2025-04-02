import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# from sklearn.preprocessing import LabelEncoder
np.set_printoptions(threshold = np.inf)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data_set = pd.read_csv('/content/all_gestures_all_users.csv')
user_id = data_set['user_id']
df = pd.DataFrame(data_set)
# df['gestures'] = df['gestures'].astype(int)

# df['new_gestures'] = df['gesture'] - 1                 #make a new column new_gestures which has values ranging from 0-5 instead of 1-6

# df

# df

array = []
array_f1=[]                                                                             #array to store accuracies

for user_id in range(1,86):

  print(f"processing user - {user_id}")
  # df = pd.read_csv('/content/processed_emg_data.csv')
  data_subset_train = df[df['user_id'] != user_id]                                        #exclude the training examples which are matching with the user_id number and the remaining data becomes the training set
  data_subset_test = df[df['user_id'] == user_id]
  # print(data_subset_train)                                      #include the training examples which are matching with the user_id number and that data becomes the test set
  X_train = data_subset_train.drop(['set','gesture','user_id','gestureName'], axis = 'columns')
  X_test = data_subset_test.drop(['set','gesture','user_id','gestureName'], axis = 'columns')        #include only the features for training


  Y_train = data_subset_train['gesture']                                                             #include only the labels for testing
  Y_test = data_subset_test['gesture']


  model = xgb.XGBClassifier(random_state = 3)                                                 #training and evaluating the model
  model.fit(X_train,Y_train)

  X_train_prediction = model.predict(X_train)

  X_test_prediction = model.predict(X_test)
  test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

  f1 = f1_score(X_test_prediction, Y_test, average = 'macro')
  array_f1.append(f1)

  train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

  test_data_accuracy = 100*test_data_accuracy

  array.append(test_data_accuracy)                                                            #calculating accuracy values

  print('Test accuracy = ',test_data_accuracy)
  print('train accuracy = ',train_data_accuracy)
  print('f1 score = ',f1)
  print()


print(array)
print(array_f1)




import matplotlib.pyplot as plt

# Assuming array contains the accuracy scores for all users except user 34

# Renumbered User IDs from 1 to 35 (skipping user 34)
user_ids_renumbered = list(range(1, len(array) + 1))

# Ensure the lengths match
assert len(user_ids_renumbered) == len(array), "Mismatch between user_ids and accuracy array lengths"

# Plot bar graph
plt.figure(figsize=(14, 6))  # Adjust figure size if needed
bars = plt.bar(user_ids_renumbered, array, color='skyblue')  # Customize color if needed

# Add text labels for accuracy scores
for bar, score in zip(bars, array):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.0f}',
             ha='center', va='bottom', fontsize=6)

# Configure plot labels and appearance
plt.xlabel('User ID')
plt.ylabel('Accuracy Score')
plt.title('Bar Graph of Accuracy Scores by User ID (Generalized Model)')
plt.xticks(user_ids_renumbered, rotation=45, fontsize=6)  # Rotate x-axis labels

# Add grid and adjust layout
plt.grid(axis='y')  # Show grid lines on the y-axis
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Display the plot
plt.show()

mean_accuracy = np.mean(array)
std_dev = np.std(array)
f1_mean = np.mean(array_f1)
f1_std_dev = np.std(array_f1)


print('mean= =',mean_accuracy)
print('std_dev =',std_dev)
print('f1_mean =',f1_mean)
print('f1_std_dev =',f1_std_dev)