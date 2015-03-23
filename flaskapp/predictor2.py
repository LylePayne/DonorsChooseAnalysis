import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import math 

#---------- MODEL IN MEMORY ----------------#

# Read the CEX data from a CSV file
# Build a LogisticRegression predictor with data
kids_data = pd.read_csv("flask_data.csv")
#kids_data.columns=['KIDS','PERSINSCQ','FDHOME','UTILCQ']

print kids_data.head()

##Create a function to take a log of each x value
def logfunction(x):
    return math.log(x+1)

##Create dataframes for x & y values to input into model
total_X = pd.DataFrame()
total_X = kids_data[['PERINSCQ','FDHOMECQ','UTILCQ']]
X = total_X.applymap(logfunction)

Y = kids_data['KIDS']
PREDICTOR = LogisticRegression().fit(X,Y)

# print X
# print Y


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """ Homepage: serve our visualization page, awesome.html
    """
    with open("Kid_Predictor2.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """  When A POST request with json data is made to this uri,
         Read the example from the json, predict probability and
         send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    new_x = np.log(x+1)
    score = PREDICTOR.predict_proba(new_x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=80)

