import json
import numpy as np
import pandas as pd
import business
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag, word_tokenize
# from glove_embedding import GloveEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import utils
from utils import handle_text
from embedding_manager_cyd import EmbeddingManagerCyd
from embedding_manager_cyd import Embedding_Type

SEED = 222
np.random.seed(SEED)

dic_business_id = {}

def main():

    embeddingManagerCyd = EmbeddingManagerCyd()

    file_path = '../data/all_data.csv'

    df = pd.read_csv(file_path)
    df['tag']=df['rating'].apply(lambda x : 1 if x >=4 else (0 if x < 3 else None))
    df1=df.dropna()
    tags=df1['tag'].to_list
    reviewList=df.reviews.apply(lambda x : embeddingManagerCyd.getEmbedding(str(x), Embedding_Type.glove, True, False).to_list


    # glove_embedding = GloveEmbedding()
    # # gloveVectors = [glove_embedding.getSentenceVectorCommon(item[1], isUseAveragePooling=True) for item in tokenizedWords.items()]
    # gloveVectors = [glove_embedding.getSentenceVectorCommon(item, isUseAveragePooling=True) for item in
    #                 reviewList]
    # features = np.array(gloveVectors, dtype=np.float16)

    features = np.array(reviewList, dtype=np.float16)
    tags = np.array(tags)

    classification_svm(features, tags)
    # classification_logistic(features, tags)

    # print("step4=================")

def get_train_test(features,tags,test_size=0.3):
    return train_test_split(features, tags, test_size=test_size, random_state=SEED)

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def classification_logistic(features, tags):
    xtrain, xtest, ytrain, ytest = get_train_test(features, tags)

    # cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)

    # lr = LogisticRegression(penalty = "l1")
    #
    # # params = {"penalty":["l1","l2"],
    # #              "C":[0.1,1.0,10.0,100.0]},
    #
    # params = {"C":[100, 120,150]},

    # grid = GridSearchCV(estimator=lr, param_grid = params)
    # grid.fit(xtrain, ytrain)
    # print("最优参数为：",grid.best_params_)
    # model = grid.best_estimator_
    # predict_value = model.predict(xtest)
    # proba_value = model.predict_proba(xtest)
    # p = proba_value[:,1]
    # print("Logistic=========== ROC-AUC score: %.3f" % roc_auc_score(ytest, p))
    #
    # joblib.dump(model, 'model/logistic_clf.pkl')

    model = LogisticRegression(penalty="l1",C = 100, solver='liblinear')
    model.fit(xtrain, ytrain)
    predict_value = model.predict(xtest)
    proba_value = model.predict_proba(xtest)
    p = proba_value[:,1]
    print("Logistic=========== ROC-AUC score: %.3f" % roc_auc_score(ytest, p))
    joblib.dump(model, 'model/logistic_clf.pkl')


def classification_svm(features, tags):
    xtrain, xtest, ytrain, ytest = get_train_test(features, tags)

    # svc = SVC(kernel='rbf',probability = True)
    # model = make_pipeline(svc)


    # # param_grid = {'svc__C': [1, 5, 10,0.5,20,30],
    # #               'svc__gamma': [0.0001, 0.0005, 0.001, 0.005,0.006,0.007,0.008,0.009,0.01]}
    # param_grid = {'svc__C': [1, 5, 10, 15],
    #               'svc__gamma': [0.0005, 0.001, 0.005]}
    # grid = GridSearchCV(model, param_grid)
    #
    # grid.fit(xtrain, ytrain)
    # print("Optimal parameters：",grid.best_params_)
    # model = grid.best_estimator_
    # predict_value = model.predict(xtest)
    # proba_value = model.predict_proba(xtest)
    # p = proba_value[:,1]
    # print("SVM=========== ROC-AUC score: %.3f" % roc_auc_score(ytest, p))

    model =SVC(kernel='rbf',probability = True, C=20, gamma=0.005)
    model.fit(xtrain, ytrain)
    proba_value = model.predict_proba(xtest)
    p = proba_value[:, 1]
    print("SVM=========== ROC-AUC score: %.3f" % roc_auc_score(ytest, p))
    joblib.dump(model, 'model/svm_clf.pkl')


if __name__ == "__main__":
    main()

