import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from scipy import stats
from scipy.stats import mstats , skew, boxcox 
import matplotlib.pyplot as plt

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

def outlier_zscore(df,column):
    """
    Pass the Dataframe and column to impute as string
    \nExample below:\n
    df['HeightInMeters']=Preproccessing.outlier_zscore(df,'HeightInMeters')
    """
    z_scores = np.abs(stats.zscore(df[column]))
    df[column] = df[column][z_scores < 3]
    return df[column]

def outlier_iqr(df,column):
    """
    Pass the Dataframe and column to impute as string
    \nExample below:\n
    df['HeightInMeters']=Preproccessing.outlier_iqr(df,'HeightInMeters')
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows within the IQR range
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered


def remove_outliers(df, column, threshold=3):
    """
    Pass the Dataframe and column to impute as string
    \nExample below:\n
    df['HeightInMeters']=Preproccessing.remove_outliers(df,'HeightInMeters')
    """
    z_scores = np.abs(stats.zscore(df[column]))
    df_cleaned = df[z_scores < threshold]
    return df_cleaned

def standard_scaler(df, column):
    """
    Standardize a column using StandardScaler (mean=0, std=1).
    Pass the DataFrame and column name as a string.
    Example:
    df['HeightInMeters'] = Preproccessing.standard_scaler(df, 'HeightInMeters')
    """
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(df[[column]])
    return pd.DataFrame(scaled_column, columns=[column], index=df.index)

def min_max_scaler(df,feature_range=(0, 1)):
    """
    Scale a column using MinMaxScaler (default range: 0 to 1).
    Pass the DataFrame, column name as a string, and optional feature range.
    Example:
    df['HeightInMeters'] = Preproccessing.min_max_scaler(df, 'HeightInMeters', feature_range=(0, 1))
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    numeric_cols = df.select_dtypes(include='number').columns
    scaled_data = scaler.fit_transform(df[numeric_cols])
    scaled_df = df.copy()
    scaled_df[numeric_cols] = scaled_data
    
    return scaled_df


def clipping(df):
    clipped_df = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        clipped_df[col] = np.clip(df[col], min_val, max_val)

    return clipped_df

def winsorization(df,low_p=5,upper_p=95):
    # Create a copy to avoid modifying the original DataFrame
    winsorized_df = df.copy()

    # Apply Winsorization to each numeric column
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        lower_limit = np.percentile(df[col], low_p)
        upper_limit = np.percentile(df[col], upper_p)

        # Clip values
        winsorized_df[col] = np.clip(df[col], lower_limit, upper_limit)

    return winsorized_df


def skewness(df):
    numeric_cols = df.select_dtypes(include='number')
    skewness_values = numeric_cols.apply(lambda x: skew(x.dropna()))

    return pd.DataFrame({'Column': skewness_values.index, 'Skewness': skewness_values.values})

def log_transform(df, columns,const=1):
    for column in columns:
        df[f'log_{column}'] = df[column].apply(lambda x: np.log(x + const))
    
    return df


def boxcox_transform(df, columns=None, shift_constant=1e-3, plot=True):
    
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    for col in columns:
        col_data = df[col]

        # Shift if data has non-positive values
        if (col_data <= 0).any():
            print(f"Column '{col}' contains non-positive values. Shifting by {shift_constant}.")
            col_data = col_data + abs(col_data.min()) + shift_constant

        try:
            transformed_data, _ = boxcox(col_data)
            df[f'boxcox_{col}'] = transformed_data

            # Plotting
            if plot:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].hist(col_data, bins=20, color='skyblue', edgecolor='black')
                axes[0].set_title(f'Original: {col}')
                axes[1].hist(transformed_data, bins=20, color='lightgreen', edgecolor='black')
                axes[1].set_title(f'Box-Cox Transformed: boxcox_{col}')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Could not apply Box-Cox to column '{col}': {e}")

    return df


    
    

    