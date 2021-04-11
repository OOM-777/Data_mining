from sklearn.preprocessing import MinMaxScaler

sacle_matrix = IFL.iloc[:, 1:4]
model_scaler = MinMaxScaler()
data_scaled = model_scaler.fit_transform(sacle_matrix)
print(data_scaled.round(2))
