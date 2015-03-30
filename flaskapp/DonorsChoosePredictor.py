import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import urlparse 
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from textblob import TextBlob
import pickle

#---------- MODEL IN MEMORY ----------------#

# Read the DonorsChoose Data from a formatted CSV table

PREDICTOR = joblib.load("GBC_Model.pkl")

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)


#Homepage
@app.route("/")
def viz_page():
    """ Homepage: serve visualization page, DonorsChooseApp.html"""
    return flask.render_template("DonorsChooseApp4.html")
        

# Get an example and return it's score from the predictor model
@app.route("/DonorsChooseResultsPage", methods = ["POST"])
def text_page():

    query = urlparse.parse_qs(flask.request.get_data())
    TeacherName = query["TeacherName"][0]
    TeacherType = query["teacher_type"][0]
    ProjectTitle = query["ProjectTitle"][0]
    NeedStatement = query["NeedStatement"][0]
    ShortDescription = query["ShortDescription"][0]
    SchoolType = query["school_type"][0]
    PovertyLevel = query["povertylevel"][0]
    TownDemo = query["towndemo"][0]
    StudentsReached = query["studentsreached"][0]
    FundingRequested = query["fundingrequested"][0]
    PrimaryFocusArea = query["subjectarea"][0]
    ResourceType = query["resource_type"][0]
    # PrimaryFocusSubject = query["focussubject"][0]

    #Additional Processing:

    #TeacherTypes:
    teacher_string = "teacher_"
    edited_teacher = teacher_string + TeacherType.replace(" ", "_").lower()

    #SchoolTypes:
    school_string = "school_"
    edited_school = school_string + SchoolType.replace(" ", "_").lower()

    #Text Lengths:
    title_length = len(ProjectTitle)
    need_length = len(NeedStatement)
    description_length = len(ShortDescription)

    #Text Sentiment and Polarity:
    TitleBlob = TextBlob(ProjectTitle)
    titlepolarity, titlesubjectivity = TitleBlob.sentiment
    NeedBlob = TextBlob(NeedStatement)
    needpolarity, needsubjectivity = NeedBlob.sentiment
    DescriptionBlob = TextBlob(ShortDescription)
    descpolarity, descsubjectivity = DescriptionBlob.sentiment

    #Need lowercase version of poverty level
    lowerPovertyLevel = str(PovertyLevel.lower())

    #Need Lowercase version of TownDemo
    lowerTownDemo = str(TownDemo.lower())



    #Importing empty DF to use for example vector:
    with open("x_df.pkl", "r") as picklefile:
        x_df = pickle.load(picklefile)
        
        if TeacherType != "None":
            x_df.loc[0, edited_teacher] = 1
        if SchoolType != "Traditional" and SchoolType != "Magnet":
            x_df.loc[0, edited_school] = 1
        x_df.loc[0, "TitleLength"] = title_length
        x_df.loc[0, "NeedLength"] = need_length
        x_df.loc[0, "DescriptionLength"] = description_length
        x_df.loc[0, "students_reached"] = int(StudentsReached)
        x_df.loc[0, "total_price_excluding_optional_support"] = int(FundingRequested)
        x_df.loc[0, "title polarity"] = titlepolarity
        x_df.loc[0, "short_description subjectivity"] = descpolarity
        x_df.loc[0, "need_statement polarity"] = needpolarity
        x_df.loc[0, "need_statement subjectivity"] = needsubjectivity
        if ResourceType != "Other":
            x_df.loc[0, ResourceType] = 1
        if PovertyLevel != "Moderate Poverty":
            x_df.loc[0, lowerPovertyLevel] = 1
        x_df.loc[0, lowerTownDemo] = 1
        ##All currently set to April
        x_df.loc[0, 4.0] = 1

        # x_vector = x_df

    # @app.route("/score", methods=["POST"])
        score = PREDICTOR.predict_proba(x_df)
        score_funded = score[0, 0]



    return flask.render_template("DonorsChooseResultsPage.html", 
                                TeacherName = TeacherName,
                                teacher_type = TeacherType,
                                ProjectTitle = ProjectTitle,
                                NeedStatement = NeedStatement,
                                ShortDescription = ShortDescription,
                                SchoolType = SchoolType,
                                PovertyLevel = PovertyLevel,
                                TownDemo = TownDemo,
                                StudentsReached = StudentsReached,
                                FundingRequested = FundingRequested,
                                PrimaryFocusArea = PrimaryFocusArea,
                                ResourceType = ResourceType,
                                X_DF = x_df,
                                Results = score_funded)
 
#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
if __name__ =='__main__':

    app.run(host='0.0.0.0', debug=True)

