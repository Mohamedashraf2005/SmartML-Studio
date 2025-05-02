import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

def labelencoding(df,column):
    """
    Pass the Dataframe and column as string
    \nExample below:\n
    df['Sex']=encoding.labelencoding(df,'Sex')
    """
    df[column] = df[column].astype('category')
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df[column]


def onehotencoding(df,column):
    """
    Pass the Dataframe and column as string
    \nExmaple below:\n
    df=encoding.onehotencoding(df,'Sex')
    """
    encoder = OneHotEncoder(sparse_output=False)
    status_encoded = encoder.fit_transform(df[[column]])
    status_columns = encoder.get_feature_names_out([column])
    status_encoded_df = pd.DataFrame(status_encoded, columns=status_columns)
    df = pd.concat([df, status_encoded_df], axis=1)
    return df.drop(columns=[column])

def imputer_mean(df,column):
    """
    Pass the Dataframe and column to impute as string
    \nExample below:\n
    df['HeightInMeters']=Preproccessing.imputer_mean(df,'HeightInMeters')
    """
    imputer = SimpleImputer(strategy='mean')
    df[column] = imputer.fit_transform(df[[column]])
    return df[column]

def imputer_fillzero(df,column):
    """
    Pass the Dataframe and column to impute as string
    \nExample below:\n
    df['HeightInMeters']=Preproccessing.imputer_fillzero(df,'HeightInMeters')
    """
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df[column] = imputer.fit_transform(df[[column]])
    return df[column]

def imputer_KNN(df,column,Neighbor):
    """
    Pass the Dataframe and column to impute as string and Neighbor as int
    \nExample below"\n
    df['HeightInMeters']=Preproccessing.imputer_KNN(df,'HeightInMeters',2)
    """
    knn_imputer = KNNImputer(n_neighbors=Neighbor)
    df[column]= knn_imputer.fit_transform(df[[column]])
    return df[column]

def imputer_Iterative(df, max_iter=10, random_state=0):    
    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(include='object').columns
    df_numeric = df_copy.drop(columns=cat_cols)
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    df_imputed = imputer.fit_transform(df_numeric)
    df_copy[df_numeric.columns] = pd.DataFrame(df_imputed, columns=df_numeric.columns, index=df.index)

    return df_copy

# What next? Standarization ,Outlier