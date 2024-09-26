import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
data = pd.read_csv('titanic.csv')
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
features = ['Pclass', 'Sex', 'Age']
target = 'Survived'
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
