import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import copy


data_set = pd.read_csv('/content/EMG-EPN.csv')
user_id = data_set['user_id']
df = pd.DataFrame(data_set)
# df.reset_index(inplace=True)

# df['new_gesture'] = df['gesture'] - 1


# from sklearn.model_selection import train_test_split

# # Define a variable to store the total count of gesture-0 in the training set
# initial_total_gesture_0_count = 0

# for user_id in range(1,30):
#     data_subset = df[df['user_id'] == user_id]

#     X = data_subset.drop(['user_id','gesture'], axis='columns')
#     Y = data_subset['gesture']

#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

#     # Count the occurrences of gesture-0 in the training set and add it to the total count
#     initial_total_gesture_0_count += sum(Y_train == 4)

# print("Total number of instances of gesture-0 in the training set:", initial_total_gesture_0_count)


user_id = data_set['user_id']
df = pd.DataFrame(data_set)
df.reset_index(inplace=True)
df['index'] = df.index + 1  # Adding 1 to start the index from 1

# df['new_gesture'] = df['gesture'] - 1

training_set = pd.DataFrame()  # Initialize an empty DataFrame
# final_total_gesture_0_count = 0

i = 0

for user_id in range(1,307):
    # if user_id == 34:
    #   continue
    data_subset = df[df['user_id'] == user_id]  # Filter data for each user

    grouped_df = data_subset.groupby('gesture')

    dfs = []
    
    for gesture, group in grouped_df:
        num_rows = int(len(group) * 0.40)
        selected_rows = group.head(num_rows)
        dfs.append(selected_rows)

    df_user_i_10_percent = pd.concat(dfs)

    training_set = pd.concat([training_set, df_user_i_10_percent])  # Concatenate to the training set DataFrame

training_set = training_set.sort_values(by='index')


training_labels = training_set['gesture'].values  # Extract labels as numpy array
# final_total_gesture_0_count = sum(training_labels == 4)
# print("Total number of instances of gesture-0 in the final training set:", final_total_gesture_0_count)


training_set = training_set.drop(['gesture','user_id','index','gestureName'], axis='columns')  # Drop unnecessary columns

model = xgb.XGBClassifier()
model.fit(training_set, training_labels)

user_id=0
#Model has been trained for the first and initial 40 percent data each from 36 users

arr_retrain_count = []
arr_accuracy = []
arr_savings = []
arr_f1 = []
overall_confusion_matrix = None  # Initialize overall confusion matrix
number_of_gesture_0_user_used_for_retraining = 0

for user_id in range(1,307):
  # if user_id == 34:
  #     continue
  data_user_i = df[df['user_id'] == user_id]
  #data_user_i has all data of i_th user

  grouped_df = data_user_i.groupby('gesture')

  grouped_data = data_user_i.groupby('gesture')

  re_training_set = pd.DataFrame()
  test_set = pd.DataFrame()

    # Split data into validation (40%) and testing (20%) for each gesture
  for gesture, group in grouped_data:
      size = int(len(group) * 0.40)
      gesture_retraining_and_test = group.iloc[size:]  # remaining data
      size_1 = int(len(group)*0.40)
      gesture_retraining = gesture_retraining_and_test.head(size_1)  # next 40% data
      gesture_test = gesture_retraining_and_test.tail(int(len(group) * 0.20))

      re_training_set = pd.concat([re_training_set, gesture_retraining])
      test_set = pd.concat([test_set, gesture_test])

  #test set should contain last 20% of the data
  print('len of re_training_set=',len(re_training_set))
  print('len of test_set=',len(test_set))





  def evaluate_and_retrain(model, user_data, user_labels, threshold=0.9):
      # retrained = False
      count_gesture_0 = 0
      for sample_data, sample_label in zip(user_data, user_labels):                 #check for each training sample in the new data
          confidence = model.predict_proba([sample_data])
          max_confidence = np.max(confidence)
          if max_confidence < threshold:
              print("Confidence below threshold. Retraining...")
              # if(sample_label==4):
              count_gesture_0 += 1
              # count_gesture_0 += 1
              X_train = np.concatenate([training_set.values])
              # y_train = np.concatenate(training_labels)
              X_train = np.concatenate([X_train, [sample_data]])
              y_train = np.concatenate([training_labels, [sample_label]])
              model.fit(X_train, y_train)
              # retrained = True

      print((f"Total retrains: {count_gesture_0}"))
      return count_gesture_0

  re_training_set = re_training_set.sort_values(by= 'index')
  new_user_labels = re_training_set['gesture'].values
  new_user_data = re_training_set.drop(['gesture','user_id','index','gestureName'], axis=1).values


  model_copy = copy.deepcopy(model)  # Create a deep copy of the model
  confidence = evaluate_and_retrain(model_copy, new_user_data, new_user_labels)
  print(f"User {user_id}: Confidence = {confidence}")
  i += confidence
  #confidence has retrain count
  #Initially We were using 80% data of user 1
  #We are using 40% of data + re_train count

  savings_gesture_0 = ((0.80*(len(data_user_i))-((0.40*(len(data_user_i)))+confidence))/(0.80*(len(data_user_i))))*100
  # savings_gesture_0 = (((initial_total_gesture_0_count)-(final_total_gesture_0_count+confidence))/(initial_total_gesture_0_count))*100



  test_labels = test_set['gesture'].values
  test_set = test_set.drop(['gesture','user_id','index','gestureName'], axis='columns')



  y_predict = model_copy.predict(test_set)
  accuracy = accuracy_score(test_labels, y_predict)


  f1 = f1_score(test_labels, y_predict, average = 'macro')


  print(accuracy)
  print(savings_gesture_0)
  print(f1)
  arr_accuracy.append(accuracy)
  arr_savings.append(savings_gesture_0)
  arr_f1.append(f1)
  arr_retrain_count.append(confidence)
  print(arr_retrain_count)

  user_confusion_matrix = confusion_matrix(test_labels, y_predict)

  # Update overall confusion matrix
  if overall_confusion_matrix is None:
      overall_confusion_matrix = user_confusion_matrix
  else:
      overall_confusion_matrix += user_confusion_matrix

mean_accuracy = np.mean(arr_accuracy*100)
mean_savings = np.mean(arr_savings*100)
mean_f1 = np.mean(arr_f1*100)
mean_retrain_count = np.mean(arr_retrain_count)


print('mean_accuracy = ', mean_accuracy*100)
print('mean_savings_gesture_i = ',mean_savings)
print('mean_f1 = ',mean_f1*100)
print('mean_retrain_count',mean_retrain_count)
# print(overall_confusion_matrix)


# print("Total number of instances of gesture-0 in the initial training set:", initial_total_gesture_0_count)
# print("Total number of instances of gesture-0 in the final training set:", final_total_gesture_0_count)
# print("Total number of instances where gesture-0 is used for retraining set:", confidence)
# print('i=',i)
# over_all_savings_gesture_0 = (((initial_total_gesture_0_count)-(final_total_gesture_0_count+i))/(initial_total_gesture_0_count))*100
# print('over_all_savings=', over_all_savings_gesture_0)


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

print('sum of enteries =',np.sum(overall_confusion_matrix))