import warnings
import sklearn
import pandas as pd
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



def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


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

    col_to_drop = ['id', 'name', 'zipcode', 'thumbnail_url', 'description', 'neighbourhood']
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

    global to_drop
    
    if not test_data:
        cor_matrix = data.corr()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.8)]
        
    data = data.drop(to_drop, axis=1)
             
    data = data.reindex(sorted(data.columns), axis=1)     
    
    #print(list(data.columns))
    
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
    
    print(get_feature_names(pipeline))
    
    return data
