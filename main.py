# app main
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
# title
st.title("Mental Health Check Web APP")

# sidebar
st.sidebar.subheader("Choose Model/ Visualization Techniques")

uploaded_file = st.file_uploader(label="upload your csv/excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("The dimensions of uploaded CSV are :", df.shape)
    except Exception as e:
        df = pd.read_excel(uploaded_file)
        st.write("The dimensions of uploaded CSV are :", df.shape)

df.isnull().sum().sort_values(ascending=False)
df.dropna(inplace=True)

column = st.sidebar.multiselect("Choose columns to drop", df.columns)
NC = df.drop(column, axis=1)

S = pd.DataFrame(NC)
df1 = NC.copy()

from sklearn.preprocessing import LabelEncoder
cols = df1.columns

st.write("Updated Dataframe :")
st.dataframe(df1)

encoder = LabelEncoder()
for col in cols:
    encoder.fit(df1[col])
    df1[col] = encoder.transform(df1[col])

plt.figure(figsize = (16, 10), dpi = 100)

corr = df1.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# corr figure plot(Bar graph)
fig, ax = plt.subplots()
sns.heatmap(df1.corr(), ax=ax)
st.write(fig)


# slider
z = st.slider('Training data split value', 0.0, 1.0, 0.25)
X=NC.iloc[:,:-1].values
y=NC.iloc[:,1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=z, random_state=10)

model = st.selectbox('Model Selection', ['Linear Regression', 'Support Vector Classification',
                                         'Logistic Regression', 'K-nearestNeighbor', 'GaussianNB'])
if model == 'Linear Regression':
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    accuracy=regressor.score(X_test,y_test)
    st.write("Accuracy of model - ", accuracy)

elif model == 'Support Vector Classification':
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    svc_pred = svc.predict(X_test)
    svc_train = svc.predict(X_train)
    accuracy = accuracy_score(y_train, svc_train)
    st.write("Accuracy of model - ", accuracy)

elif model == 'Logistic Regression':
    from sklearn.linear_model import LogisticRegression
    LL = LogisticRegression()
    LL.fit(X_train, y_train)
    LL_pred = LL.predict(X_test)
    LL_train = LL.predict(X_train)
    accuracy = accuracy_score(y_train, LL_train)
    st.write("Accuracy of model Training - ", accuracy)
    accuracy2 = accuracy_score(y_test, LL_pred)

'''
elif model == 'K-nearestNeighbor':
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=27)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_train = knn.predict(X_train)
    st.write('accuracy_train = ', accuracy_score(y_train, knn_train))
    st.write('accuracy_test = ', accuracy_score(y_test, knn_pred))
'''

else:
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb.predict(X_test)
    st.write('gnb_train = ', gnb.score(X_train, y_train))
    st.write('gnb_test =', gnb.score(X_test, y_test))
