# Functions to do exploratory data analysis on a dataset
#
# M. Vallar - 09/2023
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

cmap='YlOrBr_r'

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
  sum['missing%'] = (input_data.isna().sum())/len(input_data)
  sum['uniques'] = input_data.nunique().values
  sum['count'] = input_data.count().values
  #sum['skew'] = input_data.skew().values

  if print:
    print(sum)
  return sum


def categorize_variables(df: pd.core.frame.DataFrame)-> list,list,pd.core.frame.DataFrame:
  """
  """
  num_var         = df.select_dtypes(include=['int', 'float']).columns.tolist()
  categorical_var = df.select_dtypes(exclude=['int', 'float']).columns.tolist()
  df_encoded = _encode_variables(df, categorical_var)
  return num_var, categorical_var, df_encoded

def _encode_variables(df: pd.core.frame.DataFrame, categorical_vars: list)->pd.core.frame.DataFrame:
  """
  Hidden function to encode the variables

  Args:
    df                (obj): pd.dataframe containing the total data    
    categorical_vars  (str): list of categorical variables
  Returns:  
    df_encoded (obj): pd.dataframe with the encoded variables only
  """
  from sklearn.preprocessing import LabelEncoder

  # Create a copy of the dataframe
  df_encoded = df.copy()

  # Label encode categorical columns
  label_encoders = {}
  for column in categorical_vars:
      le = LabelEncoder()
      df_encoded[column] = le.fit_transform(df[column])
      label_encoders[column] = le
  return df_encoded

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

def numerical_histogram_plot(df: pd.DataFrame, hue:str, num_var: str)->None:
  """ 
  Makes a histogram plot of every variable divided by label 

  Args:
    df  (obj) : pd.dataframe containing the total data
    hue (str) : string with the name of the label to use to hue the data    
    num_var (str): list of strings for numerical values only
  Returns:
  """
  n_rows = len(num_var)
  fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
  sns.set_palette("Set2")
  for i, col in enumerate(num_var):
    sns.histplot(x=col, data=df, hue=hue, ax=axs[i])
    axs[i].set_title(f'{col.title()} Distribution by Target (df)', fontsize=14)
    axs[i].set_xlabel(hue, fontsize=12)
    axs[i].set_ylabel(col.title(), fontsize=12)
    sns.despine()

  fig.tight_layout()
  plt.show()

def categorical_box_plot(df: pd.DataFrame, hue:str, categorical_var: str)->None:
  """ 
  Makes a histogram plot of every variable divided by label 

  Args:
    df  (obj) : pd.dataframe containing the total data
    hue (str) : string with the name of the label to use to hue the data    
    categorical_var (str): list of strings for numerical values only
  Returns:
  """
  n_rows = len(num_var)
  fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
  sns.set_palette("Set2")
  for i, col in enumerate(categorical_var):
    sns.countplot(x=col, data=df, hue=hue, ax=axs[i])
    axs[i].set_title(f'{col.title()} Distribution by Target (df)', fontsize=14)
    axs[i].set_xlabel(hue, fontsize=12)
    axs[i].set_ylabel(col.title(), fontsize=12)
    sns.despine()

  fig.tight_layout()
  plt.show()


def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
    """
    Plot of the labels with a pie chart and a bar chart showing how many samples are in each label

    Args:
      df         (obj): pd.dataframe containing the total data    
      col        (str): column with the name of the target
      title_name (str): name of the plot
    Returns:
    """
    # Set background color
    plt.rcParams['figure.facecolor'] = '#FFFAF0'
    
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#9E3F00', '#eb5e00', '#ff781f']
    inner_colors = ['#ff6905', '#ff8838', '#ffa66b']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='YlOrBr_r', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='black', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()


def correlation_matrix_heatmap(df: pd.core.frame.DataFrame)->None:
  """
  Plot of the correlation matrix

  Args:
    df         (obj): pd.dataframe containing the total data    
  Returns:
  """
  num_var, categorical_var, df_encoded = categorize_variables(df)
  correlation_matrix_numerical(df, num_var)
  correlation_matrix_categorical(df_encoded)

def correlation_matrix_numerical(df: pd.core.frame.DataFrame, num_var: str, title_name: str='Correlation Matrix Numerical') -> None:
  """
  Plot of the correlation matrix for numerical variables

  Args:
    df         (obj): pd.dataframe containing the total data    
    num_var    (str): list of numerical variables in the dataset
    title_name (str): name of the plot
  Returns:
  """
  corr_matrix = df[num_var].corr()
  mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

  plt.figure(figsize=(15, 12))
  sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, fmt='.2f', linewidths=1, square=True, annot_kws={"size": 9} )
  plt.title('Correlation Matrix', fontsize=15)
  plt.show()

def correlation_matrix_categorical(df_encoded: pd.core.frame.DataFrame, title_name: str='Correlation Matrix categorical') -> None:
  """
  Plot of the correlation matrix for categorical variables

  Args:
    df_encoded (obj): pd.dataframe containing the encoded data    
    title_name (str): name of the plot
  Returns:
  """
  excluded_columns = ['']
  columns_without_excluded = [col for col in df_encoded.columns if col not in excluded_columns]
  corr = df_encoded[columns_without_excluded].corr()
  
  fig, axes = plt.subplots(figsize=(14, 10))
  mask = np.zeros_like(corr)
  mask[np.triu_indices_from(mask)] = True
  sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, annot=True, annot_kws={"size": 6})
  plt.title(title_name)
  plt.show()


def rescale_dataset(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
  """
  Rescale the input dataframe with the minmax method

  Args:
    df  (obj): pd.dataframe containing the total data
  Returns:
    df_resc  (obj): pd.dataframe containing the rescaled data
  """
  from mlxtend.preprocessing import minmax_scaling
  df_scaled = minmax_scaling(df, columns=df.columns.values)
  return df_scaled