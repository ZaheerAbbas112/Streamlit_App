
# import the libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# to add the heading
st.write("""
         # Explore Different ML Models and Datasets
         Which model is best than others?
         """)

# to add the names of datasets in a box by adding sidebar
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# to add the names of classifiers in a box by adding sidebar
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

# define a function to load dataset
def get_dataset (dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X,y

# call that function, and put equal to X,y variables
X, y = get_dataset(dataset_name)

# to print the shape of data set on app
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

# to add the different classifier's parameter in user input
def add_parameter_ui(classifier_name): # ui is for user input
    params = dict() # create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C # its the degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1,15)
        params['K'] = K # its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2,15)
        params['max_depth'] = max_depth # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators', 1,100)
        params['n_estimators'] = n_estimators # number of trees
    return params
        
# call the function to show the sliders
params = add_parameter_ui(classifier_name)

# make the classifier based on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name  == "SVM":
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234) # random_state, for repeat of results
    
    return clf
# now call that above finction
clf = get_classifier(classifier_name, params)

# split the data set in train and test data by(80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# now train the classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# check the model's accuracy score, and print it on app
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

### Plot Dataset ###
# draw the all features on 2 dimenssional plot by using PCA
pca = PCA(2)
X_projected = pca.fit_transform(X)

# provide the data in 0 and 1 dimenssions slice
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)

















    








