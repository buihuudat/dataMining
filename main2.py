import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file csv
data = pd.read_csv("online_shoppers_intention.csv")
# Xử lý dữ liệu
data = pd.get_dummies(data, columns=['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'])
data['Revenue'] = data['Revenue'].astype(int)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data.drop('Revenue', axis=1)
y = data['Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Huấn luyện mô hình Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
print("Accuracy of Decision Tree Classifier:", accuracy_dtc)

# Huấn luyện mô hình Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
print("Accuracy of Random Forest Classifier:", accuracy_rfc)

# huấn luyện mô hình
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# dự đoán nhãn của dữ liệu kiểm tra
y_pred = model.predict(X_test)

# Vẽ biểu đồ so sánh độ chính xác của hai mô hình trên tập kiểm tra
plt.bar(['Decision Tree', 'Random Forest'], [accuracy_dtc, accuracy_rfc])
plt.ylim(0, 1)
plt.title("Accuracy comparison between Decision Tree and Random Forest")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


