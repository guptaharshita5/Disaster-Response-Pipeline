import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response",engine) 
    X = df.filter(items=['id', 'message', 'original', 'genre'])
    y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    categories = y.columns.values
    return X,y,categories


def tokenize(text):
    
    tokens=nltk.word_tokenize(text)
    lemmatizer=nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
  
def build_model():
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
                     ])
    #X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
   # pipeline.fit(X_train['message'],Y_train)
    parameters = {'clf__max_depth': [10, 20, None],
              'clf__min_samples_leaf': [1, 2, 4],
              'clf__min_samples_split': [2, 5, 10],
              'clf__n_estimators': [10, 20, 40]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
    #cv.fit(X_train['message'], Y_train)
    return pipeline
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function for evaluating model by printing a classification report
    Args:   Model, features, labels to evaluate, and a list of categories
    Returns: Classification report
    '''
    #cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
    y_pred_test = model.predict(X_test['message'])
   # y_pred_train = model.predict(Y_test['message'])
    print(classification_report(Y_test.values, y_pred_test))
    #print('\n',classification_report(Y_train.values, y_pred_train, target_names=y.columns.values))
    #y_pred = model.predict(X_test)
    #print(classification_report(Y_test, y_pred, target_names=category_names))
    #for  idx, cat in enumerate(Y_test.columns.values):
     #   print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    #print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))

 

def save_model(model, model_filepath):
    '''
    Function for saving the model as picklefile
    Args: Model, filepath
    Returns: Nothing. Saves model to pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'],Y_train)
        
        print('Evaluating model...')
     
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()