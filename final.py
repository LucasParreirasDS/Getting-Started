import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('max_columns', None)

df = pd.read_csv('cars.csv')

df.describe()
df.info()
df.isnull().sum()

df['cubicinches'] = pd.to_numeric(df['cubicinches'], errors='coerce')
df['weightlbs'] = pd.to_numeric(df['weightlbs'], errors='coerce')

df['cubicinches'].fillna(np.mean(df['cubicinches']), inplace=True)
df['weightlbs'].fillna(np.mean(df['weightlbs']), inplace=True)

feat = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']
corr = df[feat].corr()

sns.boxplot(df.hp)

df['efic'] = 0
df['efic'] = df['mpg'].apply(lambda x: 1 if x>25 else 0)

X = df.copy()
y = X.pop('brand')

from sklearn.preprocessing import StandardScaler, LabelEncoder
encoder = LabelEncoder()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = encoder.fit_transform(y)

X[:, 3].max()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

features =  ['cylinders' ,'cubicinches' ,'hp' ,'weightlbs','time-to-60']
x13 = df[features]
y13 = df['efic']

X13_train, X13_test, y13_train, y13_test = train_test_split(x13, y13, test_size=0.30,random_state=42)

# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier(random_state=42)
# tree.fit(X13_train, y13_train)
# pred13 = tree.predict(X13_test)

from sklearn.metrics import  accuracy_score, confusion_matrix
acc = accuracy_score(y13_test, pred13)
matriz = confusion_matrix(y13_test, pred13)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=42)
log.fit(X13_train, y13_train)
pred13 = log.predict(X13_test)






# from sklearn.decomposition import PCA
# pca = PCA(n_components=7)
# princ = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)

# # Plot the explained variances
# features = range(pca.n_components_)
# plt.bar(features, pca.explained_variance_ratio_, color='black')
# plt.xlabel('PCA features')
# plt.ylabel('variance %')
# plt.xticks(features)

# PCA_components = pd.DataFrame(princ)

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3,random_state=42)

# kmeans.fit(X_train)
# pred = kmeans.predict(X_test)
# kmeans