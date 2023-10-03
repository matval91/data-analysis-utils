# This would be the steps I would follow to do some basic data analysis on the dataset
#
# M. Vallar - 10/2023
import pandas as pd
import eda

#train = pd.read_csv(whichever file)
#test = pd.read_csv(whichever file)

# label_name = Put name of the variable you want to predict

# 1 - Inspect the database
for df in [train, test]:
    df.head()
    eda.summary(df)

# 2 - Filter between numerical values and categorical values
num_var, categorical_var, train_encoded = eda.categorize_variables(train)
_, _, test_encoded = eda.categorize_variables(train)

# 3 - Check the missing data, if the step 1 showed any
eda.plot_missing_variable(train)
# We will handle the missing data on step 5

# 4 - Exploratory Data Analysis (EDA)
# 4.1 - Univariate Data Analysis
eda.numerical_histogram_plot(train, hue=label_name, num_var=num_var)
eda.categorical_box_plot(train, hue=label_name, num_var=num_var)

# 4.2 - Correlation Matrix
eda.correlation_matrix_heatmap(train)

# 4.3 - Bivariate Analysis
eda.total_violin_plot(train)

# 5 - Handle missing data
