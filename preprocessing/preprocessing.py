import joblib
import copy
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer


def transform(text):
    lex = text.split(',')
    out = []
    for item in lex:
        for el in ['{', '}', '"']:
            item = item.replace(el, '')
        for el in ['/',':',' ','-','.','&',')','(','\'']:
            item = item.replace(el,'_')
        item = item.lower()
        item = item.replace('matress','mattress')
        out.append(item)
    return out

amenitiesColumns = None

def extract_amenities(data, test_data = False):
    data_copy = data.copy()
    data_copy['amenities'] = data_copy['amenities'].apply(lambda item: transform(item))
    mlb = MultiLabelBinarizer(sparse_output=True)
    data_copy = data_copy.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(data_copy['amenities']),
            index=data.index,
            columns=mlb.classes_))
    if not test_data:
        global amenitiesColumns
        amenitiesColumns = list(mlb.classes_)
        amenitiesColumns.remove('')
        tmp = copy.deepcopy(amenitiesColumns)
    else:
        tmp = list(mlb.classes_)
        tmp.remove('')
    data_copy.drop(['amenities', ''], axis = 1, inplace = True)
    return data_copy, tmp


def preprocess_data(data, test_data = False):

    col_to_drop = ['id', 'name', 'neighbourhood', 'zipcode', 'thumbnail_url', 'description']
    data.drop(col_to_drop, axis = 1, inplace = True)

    data['host_response_rate'] = data['host_response_rate'].str.rstrip('%').astype('float')

    data.host_has_profile_pic = np.where(data.host_has_profile_pic.isnull(), 'f', data.host_has_profile_pic)
    data.host_identity_verified = np.where(data.host_identity_verified.isnull(), 'f', data.host_identity_verified)
    data['first_review'] = pd.to_datetime(data['first_review'], format='%Y-%m-%d')
    data['last_review'] = pd.to_datetime(data['last_review'], format='%Y-%m-%d')
    data['host_since'] = pd.to_datetime(data['host_since'], format='%Y-%m-%d')

    data['first_review'] = data.first_review.fillna(pd.to_datetime('2016-07-01')) # wartości najczęściej występujące
    data['last_review'] = data.last_review.fillna(pd.to_datetime('2017-09-01'))  # wartości najczęściej występujące
    data['host_since'] = data.host_since.fillna(pd.to_datetime('2015-07-01')) # wartości najczęściej występujące

    data['first_review_year'] = data['first_review'].dt.year
    data['first_review_month'] = data['first_review'].dt.month
    data.drop('first_review', axis = 1, inplace = True)

    data['last_review_year'] = data['last_review'].dt.year
    data['last_review_month'] = data['last_review'].dt.month
    data.drop('last_review', axis = 1, inplace = True)

    data['host_since_year'] = data['host_since'].dt.year
    data['host_since_month'] = data['host_since'].dt.month
    data.drop('host_since', axis = 1, inplace = True)
    
    data, pom = extract_amenities(data, test_data)

    global amenitiesColumns
    
    if test_data:
        for col in pom:
            if col not in amenitiesColumns:
                data.drop(col, axis = 1, inplace = True)
        
        for col in amenitiesColumns:
            if col not in data.columns:
                data[col] = 0
    
    print(data.isnull().sum()[data.isnull().sum() > 0] + "\n")
             
    data = data.reindex(sorted(data.columns), axis=1)     
    
    if not test_data:
        one_hot_cols = ['room_type', 'bed_type', 'city', 'property_type']
        knn_cols = ['bathrooms', 'beds', 'bedrooms', 'host_response_rate']
        ord_enc_cols = ['cancellation_policy', 'cleaning_fee', 'host_has_profile_pic', "host_identity_verified", 'instant_bookable']


        ord_enc = OrdinalEncoder(categories = [['super_strict_60', 'super_strict_30', 'strict', 'moderate', 'flexible'], 
                                            [False, True], ['f', 't'], ['f', 't'], ['f', 't']])


        pipeline = ColumnTransformer([
            ("knn", KNNImputer(), knn_cols),
            ("iter", IterativeImputer(), ['review_scores_rating']),
            ('ord_enc', ord_enc, ord_enc_cols),
            ("one_hot", OneHotEncoder(), one_hot_cols)
        ], remainder = 'passthrough')

        data = pipeline.fit_transform(data)
        
        joblib.dump(pipeline, 'custom_column_transformer.joblib')
    else:
        pipeline = joblib.load('custom_column_transformer.joblib')
        data = pipeline.transform(data)
    
    data
    
    return data