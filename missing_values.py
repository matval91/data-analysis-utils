# Functions to handle missing values on a dataset
#
# M. Vallar - 09/2023
import pandas as pd

def fill_missing_KNN_imputer(df: pd.core.frame.DataFrame, cols: list) -> pd.core.frame.DataFrame:
  """
  Fill the missing data using the KNN imputer

  Args:
    df  (obj): pd.dataframe containing the total data
    cols (list): list with name of the columns to fill
  Returns:
    df_imputed  (obj): pd.dataframe containing the imputed data
  """

  from sklearn.impute import KNNImputer
  imputer = KNNImputer(missing_values=pd.NA, n_neighbors=10)

  return pd.DataFrame(imputer.fit_transform(df), columns=cols)
