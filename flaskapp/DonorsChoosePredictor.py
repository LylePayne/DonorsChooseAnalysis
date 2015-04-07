import flask
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import urlparse 
from sklearn import preprocessing
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.externals import joblib
from textblob import TextBlob
import pickle

#---------- MODEL IN MEMORY ----------------#

# Import the Gradient Boosting Classifier Model & Scaler

PREDICTOR = joblib.load("GBCModel2.pkl")
SCALER = joblib.load("scaler.pkl")

#---------- DICTIONARY OF TERMS ----------------#

# Create dictionary to be able to access readable column headers

Label_dictionary = {'school_charter': 'School Charter',
                    'school_year_round': 'School Year Round',
                    'school_nlns': 'NLNS School',
                    'school_kipp': 'Kipp School',
                    'school_charter_ready_promise': 'Charter Ready Promise School',
                    'teacher_teach_for_america': 'Teach for America Teacher',
                    'teacher_ny_teaching_fellow': 'NY Teaching Fellow',
                    'total_price_excluding_optional_support' : 'Funding Requested',
                    'students_reached': 'Students Reached',
                    'TitleLength' : 'Title Length',
                    'DescriptionLength' : 'Description Length',
                    'NeedLength' : 'Need Length',
                    'title polarity' : 'Title Polarity',
                    'short_description subjectivity' : 'Short Description Subjectivity',
                    'need_statement polarity': 'Need Statement Polarity',
                    'need_statement subjectivity': 'Need Statement Subjectivity',
                    'high poverty' : 'High Poverty',
                    'highest poverty' : 'Highest Poverty',
                    'low poverty' : 'Low Poverty',
                    'Books' : 'Books',
                    'Supplies' : 'Supplies',
                    'Techonology' : 'Technology',
                    'Trips': 'Trips',
                    'Visitors': 'Visitors',
                    'rural': 'Rural Community',
                    'suburban': 'Suburban Community',
                    'urban' : 'Urban Community',
                    1.0 : 'January',
                    2.0 : 'February',
                    3.0 : 'March',
                    4.0 : 'April',
                    5.0 : 'May',
                    7.0 : 'July',
                    8.0 : 'August',
                    9.0 : 'September',
                    10.0: 'October',
                    11.0: 'November',
                    12.0: 'December'}

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

    #Funding Requested edited:
    Funding_string = FundingRequested.replace("$", "").replace(",", "")



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
        x_df.loc[0, "total_price_excluding_optional_support"] = float(Funding_string)
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

    # @app.route("/score", methods=["POST"])
        score = PREDICTOR.predict_proba(x_df)
        score_funded = score[0, 0]

        ##feature importances
        feature_importances = PREDICTOR.feature_importances_
        x_array = np.asarray(x_df).flatten()
        # topvalues = len(x_df) + len(feature_importances)
        # absolutes = np.absolute(x_df)

        scaled_x = SCALER.fit_transform(x_array)

        feature_values = feature_importances*scaled_x
        sorted_feature_ids = np.argsort(feature_values)
        sorted_feature_names = np.asarray(x_df.columns)[sorted_feature_ids]
        sorted_features = zip(sorted_feature_names, feature_values[sorted_feature_ids])

        most_important_feature = Label_dictionary[sorted_features[-1][0]]
        second_most_important = Label_dictionary[sorted_features[-2][0]]
        third_most_important = Label_dictionary[sorted_features[-3][0]]

        # for featurename, featureimportance in sorted_features[-:
        #     print featurename




        # topvalues = sorted_feature_values[-3:]


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
                                Results = score_funded,
                                MostImportant =  most_important_feature,
                                SecondImportant = second_most_important,
                                ThirdImportant = third_most_important)
 
#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
if __name__ =='__main__':

    app.run(host='0.0.0.0')

