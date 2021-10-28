import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xg
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from PIL import Image

st.set_page_config(layout="wide", page_title='Bank Marketing Data Science Project')

st.sidebar.title('Project Navigation')
options = st.sidebar.radio('Select a page:',
                           ['Home', 'Data Information', 'Exploratory Data Analysis', 'Model Building',
                            'Test the model'])


##--------------------------------------------------------

def home():

    st.markdown("<h1 style='text-align: center; color: black;'>Bank Marketing Project</h1>", unsafe_allow_html=True)

    image = Image.open('image.png')

    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.image(image, width=800)
    with col3:
        st.write("")

    st.subheader('This is a group project which involved each stage of a data science project lifecycle.')
    st.header('Project Contributors')
    st.subheader('1. [Suvansh Vaid](https://www.linkedin.com/in/suvanshvaid27/)')
    st.subheader('2. [Ines Perko](https://www.linkedin.com/in/ines-perko/)')
    st.subheader('3. [Zeynep Basak Eken](https://www.linkedin.com/in/zbasakeken/)')
    st.text('')
    st.subheader('Feel free to navigate through the project!')
    st.subheader('Check out the complete project on [GITHUB](https://github.com/SuvanshVaid27/Bank-Marketing-Project/tree/main/Final)')

##--------------------------------------------------------

def data_info():
    st.write(
        'The data is taken from the following UCI [repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)')

    st.header('Business Information')
    st.subheader('The Problem:')
    st.write('ABC Bank wants to sell its term deposit product to customers and before launching the product they\
             want to develop a model which helps them in understanding whether a particular customer will buy their product\
             or not (based on customer\'s past interaction with bank or other Financial Institution).')

    st.text('')
    st.subheader('The Objective:')
    st.write('The Bank wants to shortlist customers whose chances of buying the product are more so that their marketing\
            channel (tele marketing, SMS/email marketing etc) can focus only on those customers. This will save their\
            resource and their time (which is directly involved in the cost (resource billing).')

    st.text('')
    st.header('Data Information')
    st.markdown("### Input variables:\n\n\
            Bank client data:\n\
            1 - age (numeric)\n\
            2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')\n\
            3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)\n\
            4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')\n\
            5 - default: has credit in default? (categorical: 'no','yes','unknown')\n\
            6 - housing: has housing loan? (categorical: 'no','yes','unknown')\n\
            7 - loan: has personal loan? (categorical: 'no','yes','unknown')\n\n\
            Related with the last contact of the current campaign:\n\
            8 - contact: contact communication type (categorical: 'cellular','telephone')\n\
            9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')\n\
            10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')\n\
            11 - duration: last contact duration, in seconds (numeric).\n\n\
            Other attributes:\n\
            12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)\n\
            13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)\n\
            14 - previous: number of contacts performed before this campaign and for this client (numeric)\n\
            15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')\n\n\
            Social and economic context attributes:\n\
            16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)\n\
            17 - cons.price.idx: consumer price index - monthly indicator (numeric)\n\
            18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)\n\
            19 - euribor3m: euribor 3 month rate - daily indicator (numeric)\n\
            20 - nr.employed: number of employees - quarterly indicator (numeric)\n")

    st.markdown("### Output variable (desired target):\n\n\
            21 - y - has the client subscribed a term deposit? (binary: 'yes','no')\n")


##-------------------------------------------------------------------

def eda():
    data = pd.read_csv('bank-additional-full.csv', sep=';')

    st.subheader('First 5 rows of the data.')
    st.dataframe(data.head())

    categorical_features = [feature for feature in data.columns if ((data[feature].dtypes == 'O') &
                                                                    (feature not in ['y']))]
    st.subheader('View the distribution inside categorical data:')
    categorical = st.selectbox(
        'Select a categorical variable:',
        categorical_features)

    if (categorical):
        fig, ax = plt.subplots()
        ax = sns.countplot(y=str(categorical), data=data)
        plt.xlabel(str(categorical))
        plt.title('Plot of ' + str(categorical))
        st.pyplot(fig)

    st.text('')

    numerical_features = [x for x in data.columns if (x not in categorical_features) & (x != 'y')]

    st.subheader('View the boxplot distribution of numerical variables:')
    numerical = st.selectbox(
        'Select a numerical variable:',
        numerical_features)

    if (numerical):
        fig, ax = plt.subplots()
        ax = sns.boxplot(y=str(numerical), data=data)
        plt.xlabel(str(numerical))
        plt.title('Boxplot of ' + str(numerical))
        st.pyplot(fig)

    st.subheader('Now, we check the distribution of the target variable:')
    fig, ax = plt.subplots()
    ax = sns.countplot(y='y', data=data)
    plt.xlabel('y (deposit)')
    plt.title('Histogram of Target variable')
    st.pyplot(fig)

    LE = LabelEncoder()
    data['y'] = LE.fit_transform((data['y'].values))

    st.subheader('Correlation plot between all the variables:')
    cor = data.corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(cor)
    st.pyplot(fig)


##---------------------------------------------------------------

def train_predict(model, x_test, y_test):
    # make predictions on test data
    predictions = model.predict(x_test)

    st.write('Classification Report \n')

    # print classification report
    st.text(classification_report(y_test, predictions))

    st.write('AUC Score:' + str(round(roc_auc_score(y_test, predictions), 4)))


##----------------------------------------------------------------

def model():
    data = pd.read_csv('bank-additional-full.csv', sep=';')

    # encoding target label
    LE = LabelEncoder()
    data['y'] = LE.fit_transform(data.y.values)
    # encoding categorical features
    data = pd.get_dummies(data)
    X = data[[x for x in data.columns if x != 'y']]
    y = data.y
    # Train test data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write('Data split randomly!')
    st.text('')
    model_choice = st.radio('Select a model to train:', ('Logistic Regression', 'Decision Tree Classifier',
                                                         'Random Forest Classifier', 'XGBoost Classifier'))

    def train_lr(X_train, y_train):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        train_predict(lr, X_test, y_test)

    def train_dt(X_train, y_train):
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X_train, y_train)
        train_predict(clf, X_test, y_test)

    def train_rf(X_train, y_train):
        rf = RandomForestClassifier(n_estimators=1000)
        rf = rf.fit(X_train, y_train)
        train_predict(rf, X_test, y_test)

    def train_xgb(X_train, y_train):
        xgb = xg.XGBClassifier(n_estimators=100, random_state=42)
        xgb = xgb.fit(X_train, y_train)
        train_predict(xgb, X_test, y_test)

    if model_choice == 'Logistic Regression':
        train_lr(X_train, y_train)
    elif model_choice == 'Decision Tree Classifier':
        train_dt(X_train, y_train)
    elif model_choice == 'Random Forest':
        train_rf(X_train, y_train)
    else:
        train_xgb(X_train, y_train)

    # st.write('Not satisfied with the results? You must have noticed the target class is highly imbalanced!')
    # if st.button('Apply sampling techniques to solve imbalance?'):
    #
    #     if st.button('Over Sample'):
    #         ros = RandomOverSampler(random_state=42)
    #         # fit predictor and target variable
    #         x_new, y_new = ros.fit_resample(X_train, y_train)
    #         st.write('Over sampling completed!')
    #
    #     elif st.button('Under Sample'):
    #         rus = RandomUnderSampler(random_state=42)
    #         # fit predictor and target variable
    #         x_new, y_new = rus.fit_resample(X_train, y_train)
    #         st.write('Under Sampling completed')
    #
    #     elif st.button('SMOTE Sample'):
    #         smote = SMOTE()
    #         # fit predictor and target variable
    #         x_new, y_new = smote.fit_resample(X_train, y_train)
    #         st.write('SMOTE sampling completed')
    #
    #     train_again = st.button('Click to train again on the sampled data!')
    #
    #     if train_again:
    #         if model_choice == 'Logistic Regression':
    #             train_lr(x_new, y_new)
    #         elif model_choice == 'Decision Tree Classifier':
    #             train_dt(x_new, y_new)
    #         elif model_choice == 'Random Forest':
    #             train_rf(x_new, y_new)
    #         else:
    #             train_xgb(x_new, y_new)

def test():

    st.write('Welcome to the final build of this model! We have adopted the following methodology for improved results:')
    st.write('1. Model used: Random Forest Classifier')
    st.write('2. Class Imbalance handling: Under Sampling')
    st.write('3. Removed duration from the input features.')

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    data.drop(['duration'], axis=1, inplace=True)

    X = data[[x for x in data.columns if x != 'y']]
    y = data.y

    X = pd.get_dummies(X)

    LE = LabelEncoder()
    y = LE.fit_transform(y)

    st.subheader('Training the final model .........')
    rus = RandomUnderSampler(random_state=42)
    # fit predictor and target variable
    x_new, y_new = rus.fit_resample(X, y)
    rf = RandomForestClassifier(n_estimators=1000)
    rf_final = rf.fit(x_new, y_new)

    st.header('Test the model!')
    st.write('Please enter a few details about the customer and click on the Predict button to see the magic!')
    st.write('Please be patient, things may slow down (apologies for that)')

    st.subheader('Client data')
    age = st.number_input('Age **', value=30)
    job = st.selectbox('Job type **', options = data['job'].unique())
    marital = st.selectbox('Marital Status', options = data['marital'].unique())
    education = st.selectbox('Education level **', options = data['education'].unique())
    default = st.selectbox('Credit in default? **', options = data['default'].unique())
    housing = st.selectbox('Housing loan', options = data['housing'].unique())
    loan = st.selectbox('Personal Loan', options = data['loan'].unique())

    st.subheader('Data related to last contact of the current campaign')
    contact = st.selectbox('Contact Communication type **', options = data['contact'].unique())
    month = st.selectbox('Last Contact month', options = data['month'].unique())
    day_of_week = st.selectbox('Last Contact day of week', options = data['day_of_week'].unique())

    st.subheader('Other attributes')
    campaign = st.number_input('No. of contacts in the current campaign', value=30)
    pdays = st.number_input('Number of days from last contact', value=3)
    previous = st.number_input('Previous contacts performed', value=3)
    poutcome = st.selectbox('Previous campaign outcome', options = data['poutcome'].unique())

    # emp_var_rate = st.slider("Employment variation rate", -4.0,2.0)
    # cons_price_idx = st.slider("Consumer Price Index", 90.0,95.0)
    # cons_conf_idx = st.slider("Consumer Confidence Index", -50.8, -26.9)
    # euribor3m = st.slider("Euribor 3 month rate", 0.5, 6.0)
    # nr_employed = st.slider("Number of Employees", 4900, 5300)

    st.write('Note that we have input the best default values for the social indicators in the model.')

    if st.button('Predict'):

        df = pd.DataFrame({'age': age, 'job': job, 'marital': marital, 'education': education,
                           'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
                           'month': month, 'day_of_week': day_of_week, 'campaign': campaign, 'pdays': pdays,
                           'previous': previous, 'poutcome': poutcome, 'emp.var.rate': 1.0,
                           'cons.price.idx': 93.0, 'cons.conf.idx': -30.0, 'euribor3m': 0.55,
                           'nr.employed': 5100}, index=[0])

        df = pd.get_dummies(df)

        missing_cols = set(X.columns) - set(df.columns)
        for c in missing_cols:
            df[c] = 0

        result = rf_final.predict(df)

        if result == 0:
            st.warning('Oops the customer will not go for the term deposit.')
        else:
            st.warning('Yay the customer will buy the term deposit.')


##---------------------------------------------------------------

if options == 'Home':
    home()

if options == 'Exploratory Data Analysis':
    eda()

if options == 'Data Information':
    data_info()

if options == 'Model Building':
    model()

if options == 'Test the model':
    test()
