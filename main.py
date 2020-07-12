from flask import Flask
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

app = Flask(__name__)

# initiate RandomForest, download dataset and split into training and test
clf = RandomForestClassifier()
print("Downloading set....")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)#, data_home="/data/scikit") 
training_proportion = 60000/y.size # set training size
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_proportion, random_state=42)


@app.route('/<index>')
def prediction_route(index):
    return make_prediction(int(index))
 

def train_model_on_default():
    print("fitting model on default parameters...")
    clf.fit(X_train, y_train)
    return clf.score(X_test,y_test)

def train_optimized_model():
    # hyperparamater tuning
    random_grid = {'n_estimators': [10,20,50,100],
                'max_features': ['auto', 'sqrt'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]}
    print("looking for the best parameters")
    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions = random_grid,n_iter=1)
    rf_random.fit(X_train,y_train)
    print("Best params are :",rf_random.best_params_)
    clf= rf_random.best_estimator_
    return clf.score(X_test,y_test)

def make_prediction(index):
    return str(clf.predict(X_test[[index]]))

if __name__ == '__main__':
    print("The accuracy score for default parameters is ",train_model_on_default())
    #print("The accuracy score with optimized parameters is ",train_optimized_model())
    app.run(host='0.0.0.0', port=5000)