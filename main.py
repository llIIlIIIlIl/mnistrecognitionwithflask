from flask import Flask
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


clf = RandomForestClassifier()
print("Downloading set....")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)#, data_home="/app/scikit") 
training_proportion = 60000/y.size # set training size
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_proportion, random_state=42)

@app.route('/')
# TODO: ziffer anzeigen
def hello_docker():
    return 'Hello, I run in a docker container'
 

def train_model_on_default():
    print("fitting model on default parameters...")
    clf.fit(X_train, y_train)
    return clf.score(X_test,y_test)

def make_prediction(index):
    clf.predict()
    return 0


if __name__ == '__main__':
    #train  
    #app.run(host='0.0.0.0', port=5000)
    print("The accuracy score for default parameters is ",train_model_on_default())
    print("")