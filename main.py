from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


import pandas as pd

data= load_breast_cancer()

df = pd.DataFrame(data.data,columns=data.feature_names)
df["target"] = data.target

x=df[df.columns[:-1]]

y=df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred= model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")

sample_pat = pd.DataFrame([{
    "mean radius": 17.99,
    "mean texture": 10.38,
    "mean perimeter": 122.8,
    "mean area": 1001.0,
    "mean smoothness": 0.1184,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871,
    "radius error": 1.095,
    "texture error": 0.9053,
    "perimeter error": 8.589,
    "area error": 153.4,
    "smoothness error": 0.006399,
    "compactness error": 0.04904,
    "concavity error": 0.05373,
    "concave points error": 0.01587,
    "symmetry error": 0.03003,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.6,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst compactness": 0.6656,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.1189
}])

prediction = model.predict(sample_pat)

if prediction[0] == 1:
    print("Prediction: Benign")
else:
    print("Prediction: Malignant")