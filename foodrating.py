"""
A playground to test custom transformers
"""
import joblib
import matplotlib.pyplot
import datetime
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import root_mean_squared_error


import matplotlib.pyplot as plt



numeric_chars = "0123456789."


def convert_float(X):
    global e_counter
    try:
        return float(X.split(sep="/")[0].strip())
    except Exception:
        return np.nan



def csr_to_numpy(X):
    return X.toarray()


def number_converter(X):
    num_X = '0'
    if X is None:
        return 0.0
    for s in str(X):
        try:
            if s in numeric_chars:
                num_X += s
        except Exception:
            pass
    if int(num_X) == 0:
        return None
    return float(num_X)


def convert_float_2(X, func=convert_float, column="rate"):
    X_mod = X[column].apply(func=func).astype(float)
    return pd.DataFrame(data=X_mod)


def log_positive_cost(X, y=None):
    min = -np.min(X)
    pos_X = X + min
    return pos_X


def foodrating(data_asset_path=None):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    file_name = os.path.join( "datasets", "zomato.csv")
    if data_asset_path is not None: 
        rating = pd.read_csv(data_asset_path)
    else: 
        rating = pd.read_csv(file_name)
    print("In Food Rating")
    # Modify Rate column in the Dataset, to a number, to use that as the label column ultimately.
    rating = rating[rating['rate'].notnull()]
    rating["rate"] = rating["rate"].apply(func=convert_float).astype(float)
    rating = rating[rating['rate'].notnull()]  # As "rate" column is the label for the dataset, its null values are dropped rather than being imputed

    date_n = datetime.datetime.now()
    model_name = "model_z_" + "_".join((str(date_n.month) , str(date_n.day) , str(date_n.hour) , str(date_n.minute) , str(date_n.second)))
    result_name = "results_z_" + "_".join(
        (str(date_n.month), str(date_n.day), str(date_n.hour), str(date_n.minute), str(date_n.second)))

    # Create feature and label set, for training and evaluation.
    rating_label = rating["rate"].apply(lambda x:int(100*x))
    rating = rating.drop(columns=['rate'])


    # Create test and train set now, consistent so that test set does not contaminate training sets
    X_train, X_test, y_train, y_test = train_test_split(rating,
                                                        rating_label,
                                                        test_size=0.2,
                                                        random_state=256
                                                        # Ensure test and train spit do not contaminate by setting
                                                        # random state value
                                                        )

    # Pre-process the following fields
    # Fields to transform - rate, online_order, book_table,dish_liked,  rest_type, cuisines, approx_cost, listed_in(type)
    # listed_in (city)
    countv = CountVectorizer(input='content',
                             encoding='utf-8',
                             decode_error='ignore',
                             token_pattern=r'\b(\w+(?:\s+\w+)*\b)(?=\s*,|\s*$)',
                             analyzer='word'
                             )
    vectorizer_pipeline = Pipeline([
        ("impute_text", SimpleImputer(strategy="constant", fill_value="None")),
        ("reshape", FunctionTransformer(func=np.reshape, feature_names_out="one-to-one", kw_args={"newshape": -1})),
        ("count_vectorize", countv),
        ("to_array", FunctionTransformer(feature_names_out="one-to-one", func=csr_to_numpy))
    ])
    rating_trans_cl = ColumnTransformer([
        ("binary", make_pipeline(SimpleImputer(strategy="constant", fill_value="No"), OrdinalEncoder()),
         ["online_order", "book_table"]),
        ("cost", make_pipeline(FunctionTransformer(func=convert_float_2,
                                                   feature_names_out="one-to-one",
                                                   kw_args={'func': number_converter,
                                                            'column': "approx_cost(for two people)"}),
                               SimpleImputer(strategy="median"),
                               FunctionTransformer(func=np.log, feature_names_out="one-to-one"),
                               StandardScaler(with_mean=True, with_std=True)
                               ),
         ["approx_cost(for two people)"]),
        ("rest_type", vectorizer_pipeline, ["rest_type"]),
        ("cuisines", vectorizer_pipeline, ["cuisines"]),
        ("locality", OneHotEncoder(), ["listed_in(city)"])
    ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    # Create Pipeline for training
    process_rating = Pipeline(
        [
            ("preprocess", rating_trans_cl),
            ("svc", SVC(C=1.0, kernel="rbf", degree=2, gamma=0.1))
        ],
        memory=joblib.Memory()
    )
    param_grid = [
        # {"svc__C":[0.1,10,100], "svc__kernel":["rbf", "linear", "poly", "sigmoid"]},
        # {"svc__kernel":["poly"], "svc__degree":[1,2,3]},
        {"svc__C":[0.1,100], "svc__kernel":["rbf"], "svc__gamma":[0.2,10]}
    ]


    # Fit the model after preprocessing
    grd_finetune = GridSearchCV(process_rating, param_grid=param_grid, cv=2, scoring="neg_root_mean_squared_error")
    print("Starting Fit....")
    grd_finetune.fit(X_train, y_train)
    print("Finished fitting....")

    # Get resuls
    validated_score = grd_finetune.best_score_
    y_pred = grd_finetune.best_estimator_.predict(X_test)
    test_score = root_mean_squared_error(y_test, y_pred)

    # Print the results:
    print("Best Params are", grd_finetune.best_params_)
    print("Best Score ", validated_score)
    print("Test Score ", test_score)

    #dump estimator
    joblib.dump(grd_finetune.best_estimator_, os.path.join("model", model_name))


    # rating_trans_np = rating_trans_cl.fit_transform(X_train)
    # rating_preprocessed = pd.DataFrame(data=rating_trans_np, columns=rating_trans_cl.get_feature_names_out())
    # rating_preprocessed["rating__rate"] = rating_preprocessed['rating__rate'].astype(float)
    #
    #
    #
    # # Understand the data for cost for two people
    # print("Cost for two people: ", rating_preprocessed["cost__approx_cost(for two people)"].describe())
    # print(rating_preprocessed["cost__approx_cost(for two people)"])
    #
    # rating_preprocessed["cost__approx_cost(for two people)"].hist(bins=100)
    # plt.show()
    #
    # # Create training and validation data sets
    #
    #
    #

