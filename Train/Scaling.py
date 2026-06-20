from sklearn.preprocessing import StandardScaler
import pandas as pd

data = {
    'Age': [25,30,40],
    'Salary': [30000,50000,90000]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print(scaled_data)
'''[[-1.06904497 -1.06904497]
 [-0.26726124 -0.26726124]
 [ 1.33630621  1.33630621]]'''
