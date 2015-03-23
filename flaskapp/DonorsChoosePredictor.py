import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import math 
from sklearn.preprocessing import scale

#---------- MODEL IN MEMORY ----------------#

# Read the DonorsChoose Data from a formatted CSV table
# Build a LogisticRegression predictor with data
donorschoose_data = pd.read_csv("donorschoose.csv")

##Create dataframes for x & y values to input into model
X_df = pd.DataFrame()
X_df = donorschoose_data.drop("funded", 1)
X_scaled = preprocessing.scale(X_df)

Y_df = donorschoose_data['funded']
PREDICTOR = LogisticRegression().fit(X_scaled,Y_df)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """ Homepage: serve visualization page, DonorsChooseApp.html"""
    with open("DonorsChooseApp.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """  When A POST request with json data is made to this url,
         Read the example from the json, predict probability and
         send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=80)

