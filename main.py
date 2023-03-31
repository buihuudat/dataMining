import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # xây dựng mô hình cây quy
from sklearn.ensemble import RandomForestClassifier
# accuracy_score tính độ xác định của mô hình, confusion_matrix tính ma trận nhầm lẫn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# load dữ liệu
data = pd.read_csv('online_shoppers_intention.csv', sep=",", encoding='UTF-8', header=0)
print(data.head())


# Convert the Month column to numerical values
month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
data['Month'] = data['Month'].map(month_dict)

# Convert categorical data to numerical data using one-hot encoding
data = pd.get_dummies(data, columns=['VisitorType', 'Weekend', 'Month'])

data.head(10)
data.info()
data.shape  #	Hình dạng của df
data.describe() #	Mô tả df
round(100*(data.isnull().sum())/len(data),2)
data.dropna().shape

# Split the data into training and testing sets
x = data.drop(['Revenue'], axis=1)
y = data['Revenue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Fit a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Predict on the test set and calculate accuracy and confusion matrix for Decision Tree
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Fit a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set and calculate accuracy and confusion matrix for Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Print the results
print("Decision Tree Accuracy:", accuracy_dt)
print("Decision Tree Confusion Matrix:")
print(confusion_matrix_dt)
print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Confusion Matrix:")
print(confusion_matrix_rf)

# Calculate the proportion of customers who made a purchase
total_customers = len(y_test)
purchased = sum(y_test)
proportion_purchased = purchased/total_customers
print("Proportion of customers who made a purchase:", proportion_purchased)

# Dự đoán khả năng mua hàng của từng khách hàng
dt_probabilities = dt_model.predict_proba(X_test)[:,1]
rf_probabilities = rf_model.predict_proba(X_test)[:,1]

# Hiển thị kết quả tỉ lệ mua hàng của từng khách hàng
for i in range(len(dt_probabilities)):
    print(f"Customer {i+1}: Decision Tree - {dt_probabilities[i]*100:.2f}%, Random Forest - {rf_probabilities[i]*100:.2f}%")


# Tạo danh sách tên của các thuật toán và độ chính xác của chúng
algorithm_names = ['Decision Tree', 'Random Forest']
accuracy_scores = [accuracy_dt, accuracy_rf]

# Vẽ biểu đồ cột
plt.bar(algorithm_names, accuracy_scores)
plt.ylim([0.8, 1.0]) # Đặt giới hạn trục y từ 0.8 đến 1.0
plt.title('Accuracy Comparison of Decision Tree and Random Forest')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()

# Export the Decision Tree as a dot file
plt.figure(figsize=(20,10))
plot_tree(dt_model, filled=True, rounded=True, class_names=['Did not purchase', 'Purchased'], feature_names=X_train.columns)
plt.show()


from IPython.display import Image
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Export the Decision Tree as a dot file
dot_data = export_graphviz(dt_model, out_file=None,
                           feature_names=X_train.columns,
                           class_names=['Did not purchase', 'Purchased'],
                           filled=True, rounded=True,
                           special_characters=True)

# Create a graph from the dot file
graph = pydotplus.graph_from_dot_data(dot_data)

# Write the graph to a PNG file
graph.write_png('decision_tree.png')

# Display the PNG image
Image(filename='decision_tree.png')
