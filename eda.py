# Functions to do exploratory data analysis on a dataset
#
# M. Vallar - 09/2023
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

def summary(input_data: pd.DataFrame, print: bool =True)->pd.DataFrame:
  """ 
  Prints a summary of some EDA values for each row

  Args:
    input_data (obj): pd.dataframe data as input
    
  Returns:
    sum (obj): pd.dataframe output summary dataframe
  """
  sum = pd.DataFrame(input_data.dtypes, columns=['dtypes'])
  sum['missing#'] = input_data.isna().sum()
  sum['missing%'] = (input_data.isna().sum())/len(df)
  sum['uniques'] = input_data.nunique().values
  sum['count'] = input_data.count().values
  #sum['skew'] = input_data.skew().values

  if print:
    print(sum)
  return sum


def plot_pair(df: pd.DataFrame,num_var:list,target:list,
  plotname: str='Scatter Matrix with Target')->None:
  """ 
  Plots the correlation of couples of variables. The diagonal is a density function
  
  Example:
    plot_pair(df,num_var,target,plotname = 'Scatter Matrix with Target')
  Args:
    df      (obj) : pd.dataframe containing the total data
    num_var (list): a list of variables in the input dataset
    target  (list): target variable
    
  Returns:
  """
  g = sns.pairplot(data=df, x_vars=num_var, y_vars=num_var, hue=target, corner=True)
  g._legend.set_bbox_to_anchor((0.8, 0.7))
  g._legend.set_title(target)
  g._legend.loc = 'upper center'
  g._legend.get_title().set_fontsize(14)
  for item in g._legend.get_texts():
      item.set_fontsize(14)

  plt.suptitle(plotname, ha='center', fontweight='bold', fontsize=25, y=0.98)
  plt.show()

def plot_missing_variables(df: pd.DataFrame, plotname:str="Data Missing Value Matrix")->None:
  """ 
  Makes a plot of the missing values 

  Args:
    df       (obj) : pd.dataframe containing the total data    
    plotname (str) : string with the title of the plot
  Returns:
  """
  # Using package missingno to have an image of the missing values
  msno.matrix(df, color=  (0,0,0)) #color=(0.4, 0.76, 0.65)
  plt.title(plotname, fontsize=16)
  plt.show()

def total_violin_plot(df: pd.DataFrame)->None:
  """ 
  Makes a violin plot of every variable divided by target 

  Args:
    df      (obj) : pd.dataframe containing the total data    
  Returns:
  """
  cont_cols = [f for f in df.columns if df[f].dtype != 'O' and df[f].nunique() > 2]
  n_rows = len(cont_cols)
  fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
  sns.set_palette("Set2")
  for i, col in enumerate(cont_cols):
    sns.violinplot(x='outcome', y=col, data=df, ax=axs[i])
    axs[i].set_title(f'{col.title()} Distribution by Target (df)', fontsize=14)
    axs[i].set_xlabel('outcome', fontsize=12)
    axs[i].set_ylabel(col.title(), fontsize=12)
    sns.despine()

  fig.tight_layout()
  plt.show()