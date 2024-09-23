import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os

# Đọc dữ liệu
data = pd.read_csv('housing.csv')

# Xử lý dữ liệu thiếu
data.fillna({'total_bedrooms': data['total_bedrooms'].mean()}, inplace=True)

# Mã hóa cột ocean_proximity
mapping = {'<1H OCEAN': 1, 'INLAND': 2, 'NEAR OCEAN': 3, 'NEAR BAY': 4, 'ISLAND': 5}
data['ocean_proximity'] = data['ocean_proximity'].map(mapping)

# Xác định X và y
X = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'population',
          'households', 'median_income', 'ocean_proximity']]
y = data['median_house_value']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Đường dẫn để lưu mô hình
model_path = 'models/'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(model_path, exist_ok=True)

# Hàm để lưu mô hình
def save_model(model, model_name):
    joblib.dump(model, os.path.join(model_path, model_name))

# Hàm để tải mô hình
def load_model(model_name):
    return joblib.load(os.path.join(model_path, model_name))

# Kiểm tra xem mô hình đã tồn tại chưa
if os.path.exists(os.path.join(model_path, 'model_linear.joblib')):
    model_linear = load_model('model_linear.joblib')
    model_ridge = load_model('model_ridge.joblib')
    model_nn = load_model('model_nn.joblib')
    model_bagging = load_model('model_bagging.joblib')
    model_stacking = load_model('model_stacking.joblib')
else:
    # Tạo và lưu mô hình
    model_linear = LinearRegression().fit(X_train, y_train)
    save_model(model_linear, 'model_linear.joblib')
    
    model_ridge = Ridge().fit(X_train, y_train)
    save_model(model_ridge, 'model_ridge.joblib')
    
    model_nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000).fit(X_train, y_train)
    save_model(model_nn, 'model_nn.joblib')
    
    model_bagging = BaggingRegressor(estimator=model_linear, n_estimators=10).fit(X_train, y_train)
    save_model(model_bagging, 'model_bagging.joblib')
    
    estimators = [('linear', model_linear), ('ridge', model_ridge), ('mlp', model_nn)]
    model_stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge()).fit(X_train, y_train)
    save_model(model_stacking, 'model_stacking.joblib')

# Dự đoán các mô hình
predictions = {
    "Linear Regression": model_linear.predict(X_test),
    "Ridge": model_ridge.predict(X_test),
    "Neural Network": model_nn.predict(X_test),
    "Bagging Regressor": model_bagging.predict(X_test),
    "Stacking Regressor": model_stacking.predict(X_test)
}

# Tính r2 score
r2_scores = {name: r2_score(y_test, preds) for name, preds in predictions.items()}

# Tạo giao diện với Streamlit
st.title("House Price Prediction")
st.write("## Dữ liệu Housing:")
st.dataframe(data.head())

st.write("### Các thuộc tính đầu vào:")
st.write(X.columns.tolist())

st.write("### R2 Score của mô hình:")
for name, score in r2_scores.items():
    st.write(f"{name}: {score}")

# Chọn mô hình để dự đoán
model_choice = st.selectbox(
    "Chọn mô hình dự đoán:",
    list(predictions.keys())
)


# Chọn giá trị thủ công để thử dự đoán
st.write("## Dự đoán giá nhà")
longitude = st.slider("Longitude", float(X['longitude'].min()), float(X['longitude'].max()))
latitude = st.slider("Latitude", float(X['latitude'].min()), float(X['latitude'].max()))
housing_median_age = st.slider("Housing Median Age", float(X['housing_median_age'].min()), float(X['housing_median_age'].max()))
total_rooms = st.slider("Total Rooms", float(X['total_rooms'].min()), float(X['total_rooms'].max()))
population = st.slider("Population", float(X['population'].min()), float(X['population'].max()))
households = st.slider("Households", float(X['households'].min()), float(X['households'].max()))
median_income = st.slider("Median Income", float(X['median_income'].min()), float(X['median_income'].max()))
ocean_proximity = st.selectbox("Ocean Proximity", [1, 2, 3, 4, 5])

# Dự đoán với các giá trị người dùng nhập vào
input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, population, households, median_income, ocean_proximity]])



# Dự đoán dựa trên mô hình đã chọn
if(model_choice == "Linear Regression"):
    prediction = model_linear.predict(input_data)

if(model_choice == "Ridge"):
    prediction = model_ridge.predict(input_data)

if(model_choice == "Neural Network"):
    prediction = model_nn.predict(input_data)

if(model_choice == "Bagging Regressor"):
    prediction = model_bagging.predict(input_data)

if(model_choice == "Stacking Regressor"):
    prediction = model_stacking.predict(input_data)    


st.write("### Giá nhà dự đoán:")
st.write(prediction)
