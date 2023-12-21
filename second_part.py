# %% [markdown]
# ### librerias

# %%
import pandas as pd
import numpy as np
import pandas as pd

# Set the maximum number of rows and columns to display
pd.set_option('display.max_rows', 1000)  # Adjust to the number of rows you want to display
pd.set_option('display.max_columns', 1000)  # Adjust to the number of columns you want to display
import gc # TODO esto depende del estilo que sigas, pero suele ser bueno poner tus imports al comienzo del archivo
gc.collect()

# %% [markdown]
# ### Funciones

# %%
def reduce_mem_usage(df):
    """ itera todas las columnas del dataframe y modifica el tipo de dato para reducir la memoria de uso
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type=='datetime64[ns]': pass
        elif col_type not in  [object,'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

# %%
import pandas as pd
import numpy as np

# funcion para revisar los principales indicadores
def custom_describe(df):
    # conviertiendo a numerico los datos
    df = df.apply(pd.to_numeric, errors='coerce')

    summary = df.describe(percentiles=[0.1, 0.25,0.5, 0.75, 0.9, 0.99]).T

    # Calculando los numeros de ceros
    num_zeros = (df == 0).sum()

    # Calculate the percentage of zero values
    percent_zeros = (num_zeros / df.shape[0]) * 100

    # Calculate the number of missing values
    num_missing = df.isnull().sum()

    # Calculate the percentage of missing values
    percent_missing = (num_missing / df.shape[0]) * 100

    # Calculate the number of infinity (inf) values
    num_inf = (df == np.inf).sum()

    # Create a flag for the presence of inf values
    has_inf = num_inf > 0

    # Add the number of zero values and percentage to the summary
    summary['num_zeros'] = num_zeros
    summary['percent_zeros'] = percent_zeros

    # Add the number of missing values and percentage to the summary
    summary['num_missing'] = num_missing
    summary['percent_missing'] = percent_missing

    # Add the number of infinity values and a flag for their presence
    summary['num_inf'] = num_inf
    summary['has_inf'] = has_inf

    return summary

# TODO me parece que esta es la misma función de la parte 1. No es bueno duplicar código, puede llevar a errores. Así que es mejor defijirla en un archivo separado y llemarla desde esa ruta.

# %% [markdown]
# ### Cargamos la base

# %%
path = ''
df = pd.read_csv(path+'./dataset_prueba.csv')
df.head()

# %%
df = reduce_mem_usage(df)

# %%
df.shape, df['mobile_number'].nunique()

# %%
df['last_date_of_month'].value_counts()

# %%
### checking quality of data
agr = {'mobile_number':['count','nunique']}
q = df.groupby('last_date_of_month').agg(agr).reset_index()
q.columns = ['last_date_of_month', 'mobile_number_count', 'mobile_number_nunique']
q

# TODO q es un nombre de variable muy ambiguo. Es mejor que uses uno más descriptivo

# %%
df.groupby('mobile_number')['last_date_of_month'].count()

# %%
df[df['mobile_number']==7000000074]    

# TODO  los códigos que uses para explorar los datos y debugear no deberían quedar en la rama productiva

# %% [markdown]
# ### Split the data

# %%
df_train = df[df['last_date_of_month'].isin(['6/30/2014','7/31/2014'])]## data training
df_oot = df[df['last_date_of_month'].isin(['7/31/2014','8/31/2014'])] ## we still use the 07 2014 yes, for the features

# TODO tienes problemas de data leakage, porque el 7 está en ambos conjuntos

# %%
#df[df['mobile_number']==7002216683][['last_date_of_month','churn']]

# TODO Los códigos comentados no deben quedar en ninguna versión estable. Este es un error muy repetido

# %% [markdown]
# ### Feature Engineering

# %%
import datetime # TODO importas esta librería y no la usas
## parsing data
for x in df_train,df_oot: # TODO no entendí por qué hiciste un for de x en dos dataframes. x es de la dimención del segundo dataframe, para qué te sirve el otro?
    x['last_date_of_month'] = pd.to_datetime(x['last_date_of_month'])  #parsing
    x['date_of_last_rech'] = pd.to_datetime(x['date_of_last_rech'])  #parsing    
    x['date_of_last_rech_data'] = pd.to_datetime(x['date_of_last_rech_data'])  #parsing    
    
    x['day_of_last_rech']  = (x['last_date_of_month'] -x['date_of_last_rech']).dt.days   #new features
    x['day_of_last_rech_data'] = (x['last_date_of_month']-x['date_of_last_rech_data']).dt.days  #new features
    
    x['day_of_last_rech'].fillna(-1,inplace=True) #imputing
    x['day_of_last_rech_data'].fillna(-1,inplace=True) #imputing
    
    del x['date_of_last_rech']
    del x['date_of_last_rech_data']

# TODO Qué es x? Nombre ambiguo

# %%
df_train.info()

# %% [markdown]
# datetime64[ns](1), float64(45), int64(10) , data is ready without categories but we have datetime features

# %% [markdown]
# ##### creating lag of information

# %%
df_train.columns

# %%
column_features=  ['arpu', 'onnet_mou', 'offnet_mou', 'roam_ic_mou',
       'roam_og_mou', 'loc_og_t2t_mou', 'loc_og_t2m_mou', 'loc_og_t2f_mou',
       'loc_og_t2c_mou', 'loc_og_mou', 'std_og_t2t_mou', 'std_og_t2m_mou',
       'std_og_t2f_mou', 'std_og_t2c_mou', 'std_og_mou', 'isd_og_mou',
       'spl_og_mou', 'og_others', 'total_og_mou', 'loc_ic_t2t_mou',
       'loc_ic_t2m_mou', 'loc_ic_t2f_mou', 'loc_ic_mou', 'std_ic_t2t_mou',
       'std_ic_t2m_mou', 'std_ic_t2f_mou', 'std_ic_t2o_mou', 'std_ic_mou',
       'total_ic_mou', 'spl_ic_mou', 'isd_ic_mou', 'ic_others',
       'total_rech_num', 'total_rech_amt', 'max_rech_amt', 'last_day_rch_amt',
       'total_rech_data', 'max_rech_data', 'count_rech_2g', 'count_rech_3g',
       'av_rech_amt_data', 'vol_2g_mb', 'vol_3g_mb', 'arpu_3g', 'arpu_2g',
       'night_pck_user', 'monthly_2g', 'sachet_2g', 'monthly_3g', 'sachet_3g',
       'fb_user', 'day_of_last_rech']

# %%
temporal = df_train[column_features]
summary = custom_describe(temporal)
summary

# %% [markdown]
# constante features :std_og_t2c_mou & std_ic_t2o_mou

# %%
### we're finding negative values on arpu & arpu_3g & arpu2g, could be for people who don't pay their debts
df_train['arpu_2g'].hist()

### An idea of new feature could be the min of the arpu in the last months

# %%
#df_train['total_rech_data'].value_counts()

# %%
df_train['std_ic_t2o_mou'].value_counts()
### droping these variables becouse exist constant values
###  std_ic_t2o_mou & std_og_t2c_mou 

# %% [markdown]
# ##### Understanding the missing for future imputation

# %%
### % of missing values per feature
null_prc = (df_train.isnull().sum() / len(df_train)) * 100
null_prc

# %%
df_train['count_rech_3g'].value_counts()

# %%
df_train['av_rech_amt_data'].value_counts()

# %%
df_train['arpu_2g'].describe()

# %%
df_train['arpu_2g'].value_counts()

# %%
df_train['night_pck_user'].value_counts() ### imputing with 0

# %%
df_train['fb_user'].value_counts() ## imputing with 0

# %%
column_features = ['arpu', 'onnet_mou', 'offnet_mou', 'roam_ic_mou',
                   'roam_og_mou', 'loc_og_t2t_mou', 'loc_og_t2m_mou', 'loc_og_t2f_mou',
                   'loc_og_t2c_mou', 'loc_og_mou', 'std_og_t2t_mou', 'std_og_t2m_mou',
                   'std_og_t2f_mou', 'std_og_t2c_mou', 'std_og_mou', 'isd_og_mou',
                   'spl_og_mou', 'og_others', 'total_og_mou', 'loc_ic_t2t_mou',
                   'loc_ic_t2m_mou', 'loc_ic_t2f_mou', 'loc_ic_mou', 'std_ic_t2t_mou',
                   'std_ic_t2m_mou', 'std_ic_t2f_mou', 'std_ic_t2o_mou', 'std_ic_mou',
                   'total_ic_mou', 'spl_ic_mou', 'isd_ic_mou', 'ic_others',
                   'total_rech_num', 'total_rech_amt', 'max_rech_amt', 'last_day_rch_amt',
                   'total_rech_data', 'max_rech_data', 'count_rech_2g', 'count_rech_3g',
                   'av_rech_amt_data', 'vol_2g_mb', 'vol_3g_mb', 'arpu_3g', 'arpu_2g',
                   'night_pck_user', 'monthly_2g', 'sachet_2g', 'monthly_3g', 'sachet_3g',
                   'fb_user', 'day_of_last_rech']
for x in df_train,df_oot:
    for col in column_features:
        x[col].fillna(0,inplace=True)

# TODO nombre de variable ambiguo. Este es un error muy repetido

# %% [markdown]
# ##### Outliers

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Create box plots for all numeric variables
sns.set(style="whitegrid")  # Set style for Seaborn plot

# Plot box plots for each numeric variable
plt.figure(figsize=(20, 10))  # Adjust the figure size
sns.boxplot(data=df_train[column_features], orient="h")  # Create horizontal box plots
plt.title("Box Plot for Numeric Variables")  # Set the title
plt.xlabel("Value")  # Label for the x-axis
plt.ylabel("Variable")  # Label for the y-axis
plt.show()

# %% [markdown]
# it's possible that this outliers could explain the attrition of the clientes for that we'll not make a capeo or replacing the outliers

# %% [markdown]
# #### Casteo de variables

# %%
for x in df_train, df_oot:
    x['last_date_of_month'] = x['last_date_of_month'].astype(str)
    x['last_date_of_month'] = x['last_date_of_month'].str[0:7]


# TODO repites muchas veces este tipo de iteración. Podrías usar una sola iteracion

# %%
## Sort the information because the type of model,we'll modelate with july (2014-07), use the information of june (2014-06) and predict (2014-08

for x in df_train, df_oot:
    x = x.sort_values('last_date_of_month')

# %%
### Generando variables historicas (ultimo 2 meses)(junio-julio)

# %%
# List of features to aggregate
features_to_aggregate = column_features  

# Group by 'mobile_number' and aggregate the features
grouped_train = df_train.groupby('mobile_number').agg({
    feature: ['mean', 'max', 'min', 'std','median','first'] for feature in features_to_aggregate
})

# Reset the column names
grouped_train.columns = ['_'.join(col).strip() for col in grouped_train.columns.values]

# TODO Desde aquí se me comenzó a acabar el tiempo para revisar y tuve que acelerar, porque tu código era muy largo y tenía que revisarlo rápido. 

# Reset the index to have 'mobile_number' as a regular column
grouped_train = grouped_train.reset_index()
grouped_train.head()

# %%
### Generando variables historicas para la prediccion (ultimo 2 meses)(parados en agosto)

# %%
# List of features to aggregate
features_to_aggregate = column_features  

# Group by 'mobile_number' and aggregate the features
grouped_oot = df_oot.groupby('mobile_number').agg({
    feature: ['mean', 'max', 'min', 'std','median','first'] for feature in features_to_aggregate
})

# Reset the column names
grouped_oot.columns = ['_'.join(col).strip() for col in grouped_oot.columns.values]

# Reset the index to have 'mobile_number' as a regular column
grouped_oot = grouped_oot.reset_index()
grouped_oot.head()

# %%
df_oot[df_oot.mobile_number==7002398245]

# TODO muchas lineas sin utilidad. Eso agrega mucho ruido/clutter y hace más difícil la lectura

# %%
df_oot.last_date_of_month.value_counts()

# %% [markdown]
# # 1. Modelamiento por cliente, data model= 2014-07

# %%
### Base de id's unicos del mes de julio
universe_train = df_train[df_train['last_date_of_month']=='2014-07'].groupby(['mobile_number'])['churn'].max().reset_index()
universe_train.head()

# %%
### generando la base de predicción de agosto
universe_oot = df_oot[df_oot['last_date_of_month']=='2014-08'].groupby(['mobile_number'])['churn'].max().reset_index()
universe_oot.head()

# %% [markdown]
# Revisando la calida de data

# %%
universe_train['mobile_number'].nunique(), len(universe_train)

# %%
universe_oot['mobile_number'].nunique(), len(universe_oot)

# %% [markdown]
# ### Pegamos las variables generadas a las bases de modelamiento

# %%
universe_train = universe_train.merge(grouped_train, how = 'left',on = ['mobile_number'])
universe_train.shape

# %%
universe_oot = universe_oot.merge(grouped_oot, how = 'left',on = ['mobile_number'])
universe_oot.shape

# %%
universe_train.sample(1)

# %%
universe_oot.sample(1)

# %% [markdown]
# ##### Feature selection using analysis Bivariado (Gini x variable)

# %%
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def calculate_roc_auc_for_columns_binary(df, target_column, columns_of_interest):
    """
    Calculate ROC AUC scores for specified columns in a DataFrame for binary classification.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    target_column (str): Name of the binary target column (0 or 1).
    columns_of_interest (list): List of column names to calculate ROC AUC for.

    Returns:
    pd.DataFrame: DataFrame containing the ROC AUC scores for each column.
    """
    roc_auc_scores = {}

    # Calculate ROC AUC for each column
    for col in columns_of_interest:
        target = df[target_column]
        roc_auc = roc_auc_score(target, df[col])
        roc_auc_scores[col] = roc_auc

    # Convert the dictionary to a DataFrame
    roc_auc_df = pd.DataFrame(roc_auc_scores.items(), columns=["Feature", "ROC_AUC"])
    
    return roc_auc_df

# Example usage:
# Assuming your DataFrame is named 'train2' and you want to calculate ROC AUC for specific columns
target_column = 'churn'  # Change this to the name of your binary target column
columns_of_interest = universe_train.drop(columns=['mobile_number', 'churn'], axis=1).columns
roc_auc_df = calculate_roc_auc_for_columns_binary(universe_train, target_column, columns_of_interest)

# Print or use the DataFrame
roc_auc_df.head()

# %%
roc_auc_df['GINI']= roc_auc_df['ROC_AUC']*2-1

# %%
roc_auc_df.sort_values('GINI', ascending=False)

# %%
roc_auc_df[roc_auc_df['ROC_AUC']>0.5]['Feature'].unique()

# %%
columns_pass = ['arpu_max', 'arpu_std', 'onnet_mou_std', 'offnet_mou_std',
       'roam_ic_mou_mean', 'roam_ic_mou_max', 'roam_ic_mou_std',
       'roam_ic_mou_median', 'roam_og_mou_mean', 'roam_og_mou_max',
       'roam_og_mou_std', 'roam_og_mou_median', 'loc_og_mou_std',
       'std_og_t2t_mou_std', 'std_og_t2m_mou_std', 'std_og_mou_std',
       'isd_og_mou_mean', 'isd_og_mou_max', 'isd_og_mou_std',
       'isd_og_mou_median', 'og_others_mean', 'og_others_max',
       'og_others_min', 'og_others_std', 'og_others_median',
       'total_og_mou_std', 'total_ic_mou_std', 'total_rech_num_std',
       'total_rech_amt_std', 'max_rech_amt_std', 'last_day_rch_amt_std',
       'total_rech_data_std', 'max_rech_data_std', 'count_rech_3g_mean',
       'count_rech_3g_max', 'count_rech_3g_std', 'count_rech_3g_median',
       'av_rech_amt_data_std', 'monthly_3g_mean', 'monthly_3g_max',
       'monthly_3g_std', 'monthly_3g_median', 'fb_user_std',
       'day_of_last_rech_mean', 'day_of_last_rech_max',
       'day_of_last_rech_min', 'day_of_last_rech_std',
       'day_of_last_rech_median', 'day_of_last_rech_first'] 

# %% [markdown]
# ##### Feature selection Multivariado  (Random Forest Selector) 

# %%
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# %%
universe_train.head()

# %%
universe_train[columns_pass].info()

# %%
# Create a DataFrame containing only the selected columns
universe_train_selected = universe_train[columns_pass]

y_train = universe_train['churn'] 
# Initialize a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the Random Forest classifier on the selected features
rf_classifier.fit(universe_train_selected, y_train)

# Get feature importances from the trained model
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to store feature importances
importance_df = pd.DataFrame({'Feature': universe_train_selected.columns, 'Importance': feature_importances})

# Sort the DataFrame by feature importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print or use the importance DataFrame
importance_df

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Assuming you have already created 'importance_df' as in your code

# Sort the DataFrame by feature importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %%
importance_df[importance_df['Importance']>=0.005]['Feature'].nunique()

# %%
importance_df_selected_columns = ['arpu_std', 'arpu_max', 'total_ic_mou_std',
       'day_of_last_rech_first', 'offnet_mou_std', 'total_og_mou_std',
       'loc_og_mou_std', 'onnet_mou_std', 'total_rech_amt_std',
       'total_rech_num_std', 'day_of_last_rech_median',
       'day_of_last_rech_std', 'day_of_last_rech_mean', 'std_og_mou_std',
       'max_rech_amt_std', 'day_of_last_rech_min', 'day_of_last_rech_max',
       'std_og_t2m_mou_std', 'last_day_rch_amt_std', 'std_og_t2t_mou_std',
       'roam_og_mou_max', 'og_others_max', 'og_others_mean',
       'roam_og_mou_median', 'og_others_std', 'roam_og_mou_std',
       'av_rech_amt_data_std', 'roam_og_mou_mean', 'roam_ic_mou_median',
       'og_others_median', 'roam_ic_mou_mean', 'roam_ic_mou_max',
       'max_rech_data_std', 'roam_ic_mou_std', 'total_rech_data_std',
       'isd_og_mou_mean', 'isd_og_mou_std', 'isd_og_mou_median',
       'isd_og_mou_max']

# %%
importance_df_selected = importance_df[importance_df['Importance']>=0.005]

# %%
importance_df_selected

# %%
## grafica de las variables después del filtro
import matplotlib.pyplot as plt

N = 39  # Change this to the number of top features you want to plot

# Sort the DataFrame by importance in descending order and select the top N
top_features = importance_df_selected.sort_values(by='Importance', ascending=False).head(N)

# Create a bar plot for the top N features
plt.figure(figsize=(12, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Top {N} Feature Importances')
plt.gca().invert_yaxis()  # Invert the y-axis to display the most important at the top
plt.show()

# %% [markdown]
# #### Filtro de correlaciones

# %%
# Calculate the pairwise correlation matrix among the selected features
correlation_matrix = universe_train[importance_df_selected_columns+['churn']].corr()

# Identify features with correlation > 0.8 with any other feature
highly_correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            highly_correlated_pairs.append((colname_i, colname_j, correlation_matrix.iloc[i, j]))

# Print or use the highly correlated feature pairs and their correlation values
for pair in highly_correlated_pairs:
    print(f"Features '{pair[0]}' and '{pair[1]}' have correlation: {pair[2]:.2f}")

# %%


# %%
# Calculate the pairwise correlation matrix among the selected features
correlation_matrix = universe_train[importance_df_selected_columns + ['churn']].corr()

# Set the correlation threshold (e.g., 0.9)
correlation_threshold = 0.9

# Identify features with correlation > correlation_threshold with any other feature
highly_correlated_columns = set()  # Use a set to store unique column names

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            highly_correlated_columns.add(colname_i)
            highly_correlated_columns.add(colname_j)

# Convert the set of highly correlated columns to a list
highly_correlated_columns_list = list(highly_correlated_columns)

# Print or use the list of highly correlated columns
print("Highly correlated columns:", highly_correlated_columns_list)


# %%
### droping the variables with correlation

# %%
final_features = [
    'arpu_std',
    'arpu_max',
    'total_ic_mou_std',
    'day_of_last_rech_first',
    'offnet_mou_std',
    'total_og_mou_std',
    'loc_og_mou_std',
    'onnet_mou_std',
    'total_rech_amt_std',
    'total_rech_num_std',
    'day_of_last_rech_median',
    'day_of_last_rech_std',
    'std_og_mou_std',
    'max_rech_amt_std',
    'day_of_last_rech_min',
    'std_og_t2m_mou_std',
    'last_day_rch_amt_std',
    'std_og_t2t_mou_std',
    'roam_og_mou_max',
    'og_others_max',
    'av_rech_amt_data_std',
    'roam_ic_mou_median',
    'roam_ic_mou_mean',
    'max_rech_data_std',
    'roam_ic_mou_std',
    'total_rech_data_std',
    'isd_og_mou_mean',
    'isd_og_mou_std',
]

# %%
len(final_features)

# %% [markdown]
# #### Sentido de la variable con el target

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'universe_train' contains your data and the selected features

# Create quintiles for each feature
for feature in final_features:
    universe_train[feature + '_quintile'] = pd.qcut(universe_train[feature], 5, labels=False, duplicates='drop')

# Calculate the average 'churn' value for each quintile of each feature
quintile_averages = []
for feature in final_features:
    quintile_avg = universe_train.groupby(feature + '_quintile')['churn'].mean()
    quintile_averages.append(quintile_avg)

# Create subplots for each feature
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(18, 12))

for i, feature in enumerate(final_features):
    row, col = divmod(i, 6)
    ax = axes[row, col]
    ax.plot(quintile_averages[i], marker='o')
    ax.set_title(feature)
    ax.set_xlabel('Quintile')
    ax.set_ylabel('Average Churn')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

plt.tight_layout()
plt.show()

# TODO hiciste varios gráficos, pero no en todo explicaste cuál fue el objetivo y lo que aprendiste de cada gráfico. No sirve hacer gráficos por hacer gráficos, tienen que entregasrte una información clara que puedas resumir. 

# %% [markdown]
# checking the graph with only points

# %%
### a lot of 0's
universe_train['og_others_max'].value_counts()

# %% [markdown]
# ### Modelling Selector

# %%
# !pip install pycaret

# TODO Más códigos que hay que borrar

# %%
from pycaret.classification import *

# %%
universe_train['churn'].value_counts()/len(universe_train)

# %%
exp1 = setup(data=universe_train[final_features+['churn']], target='churn', session_id=123,fix_imbalance=False)

# %%
best_model = compare_models()

# %%
rf = create_model('rf')### aplicando smoothe

# %%
# Create the best model sin smoothe
rf = create_model('rf')
#rf	Random Forest Classifier	0.9782	0.8631	0.1813	0.2307	0.2026	0.1918	0.1934	20.4410

# %%
#tuned_rf = tune_model(tuned_rf, optimize='AUC', fold=5, choose_better=True, search_library='optuna', custom_grid = {"eval_metric": ['AUC', 'F1']})

# %% [markdown]
# ### Model Cliente Final 

# %%
# !pip install optuna

# %%
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
# Load your data and split it into features (X) and target (y)
X = universe_train[final_features]
y = universe_train['churn']

# %%
def objective(trial):
    # Define the hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 100, 150)
    max_depth = trial.suggest_int('max_depth', 5, 10)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)

    # Create the Random Forest classifier with the suggested hyperparameters
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf, random_state=42)

    # Perform cross-validation
    f1_scores = cross_val_score(rf, X, y, cv=5, scoring='f1')

    return np.mean(f1_scores)

# %%
study = optuna.create_study(direction='maximize')  # Maximize F1-score
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train the Random Forest model with the best hyperparameters using the full dataset
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X, y)

# %%
"""RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=9, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=0.3424342509711944,
                       min_samples_split=0.5404161634092272,
                       min_weight_fraction_leaf=0.0, n_estimators=132,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)"""

# %% [markdown]
# Entrenando el modelo con los indicadores anteriores

# %%
### simple search Mean F1 Score on Training Data (Cross-Validation): 0.09080806470731605

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

# Define the hyperparameter search space
param_dist = {
    'n_estimators': [100, 300],
    'max_depth': [-1, 10],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [2, 4]
}

# Create a Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the F1 scorer
f1_scorer = make_scorer(f1_score)

# Create a StratifiedKFold cross-validation object
stratified_cv = StratifiedKFold(n_splits=5, random_state=42,shuffle=True)

# Create a RandomizedSearchCV object with stratified cross-validation and verbose output
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   scoring=f1_scorer, n_iter=10, cv=stratified_cv, random_state=42, verbose=50)

# Fit the RandomizedSearchCV object to your data
random_search.fit(X, y)

# Get the best hyperparameters and the corresponding F1 score
best_params = random_search.best_params_
best_f1 = random_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best F1 Score:", best_f1)

# %%
best_params

# TODO Falta guardar estos outputs en archivos para poder respaldarlos. JN NO es una buena forma de guardar resultados. Es más, muchas veces no deberían guardarse los resultados en ellos. Tienes que tener una forma confiable de guartdar tus outputs 

# %% [markdown]
# ###### Revisando indicadores en la data OOT del Modelo Cliente

# %%
from sklearn.ensemble import RandomForestClassifier

# Define the hyperparameters
hyperparameters = {
    'n_estimators': 300,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_depth': 10
}

# Create a Random Forest classifier with the specified hyperparameters
rf = RandomForestClassifier(**hyperparameters, random_state=42)

# Fit the Random Forest model to your in-sample data (X, y)
rf.fit(X, y)

# Now, let's test the model on the OOT dataset (X_oot)
X_oot = universe_oot[final_features].fillna(0)

# Get predictions (0 or 1) for OOT dataset
predictions = rf.predict(X_oot)

# Get class probabilities for both classes (0 and 1)
class_probabilities = rf.predict_proba(X_oot)

# Class 1 probabilities (churn)
class_1_probabilities = class_probabilities[:, 1]

# Add the churn probabilities and predictions to your OOT dataset
universe_oot['Churn_Probabilities'] = class_1_probabilities
universe_oot['Churn_Predictions'] = predictions


# %%
universe_oot.head()

# %%
# Calculate and print F1 score on the training data
f1_train = f1_score(y, rf.predict(X))
print("\nTraining Data Metrics:")
print("F1 Score (Training):", f1_train)

# %% [markdown]
# Indicadores del Modelo en el mes de AGOSTO

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Extract actual churn values and predicted churn values from your DataFrame
actual_churn = universe_oot['churn']
predicted_churn = universe_oot['Churn_Predictions']
predicted_prob_churn = universe_oot['Churn_Probabilities']

# Calculate classification metrics
accuracy = accuracy_score(actual_churn, predicted_churn)
precision = precision_score(actual_churn, predicted_churn)
recall = recall_score(actual_churn, predicted_churn)
f1 = f1_score(actual_churn, predicted_churn)
auc = roc_auc_score(actual_churn, predicted_prob_churn)
confusion = confusion_matrix(actual_churn, predicted_churn)
confusion_df = pd.DataFrame(confusion, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)
print("Confusion Matrix:\n", confusion_df)

# %% [markdown]
# Feature importance

# %% [markdown]
# # Modelo 2  CLIENTE - MES

# %%
df_train_2 = df_train[df_train['last_date_of_month'].isin(['2014-06','2014-07'])]
df_oot_2 = df_oot[df_oot['last_date_of_month']=='2014-08']

# %%
df_train_2['last_date_of_month'].value_counts()

# %%
df_oot_2['last_date_of_month'].value_counts()

# %%
df_train_2.info() ##last_date_of_month     object type

# %%
df_train_2.isnull().sum() ## not nulls

# %%
df_train_2['churn'].value_counts()/len(df_train_2)

# %%
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def calculate_roc_auc_for_columns_binary(df, target_column, columns_of_interest):
    """
    Calculate ROC AUC scores for specified columns in a DataFrame for binary classification.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    target_column (str): Name of the binary target column (0 or 1).
    columns_of_interest (list): List of column names to calculate ROC AUC for.

    Returns:
    pd.DataFrame: DataFrame containing the ROC AUC scores for each column.
    """
    roc_auc_scores = {}

    # Calculate ROC AUC for each column
    for col in columns_of_interest:
        target = df[target_column]
        roc_auc = roc_auc_score(target, df[col])
        roc_auc_scores[col] = roc_auc

    # Convert the dictionary to a DataFrame
    roc_auc_df = pd.DataFrame(roc_auc_scores.items(), columns=["Feature", "ROC_AUC"])
    
    return roc_auc_df

# Example usage:
# Assuming your DataFrame is named 'train2' and you want to calculate ROC AUC for specific columns
target_column = 'churn'  # Change this to the name of your binary target column
columns_of_interest = df_train_2.drop(columns=['mobile_number', 'churn','last_date_of_month'], axis=1).columns
roc_auc_df = calculate_roc_auc_for_columns_binary(df_train_2, target_column, columns_of_interest)

# Print or use the DataFrame
roc_auc_df

# %%
roc_auc_df[roc_auc_df['ROC_AUC']>=0.5] ### only two variables pass, we'll try other filter selector like Chi-square

# %% [markdown]
# Information Value

# %%
import pandas as pd
from scipy.stats import chi2_contingency

def calculate_chi_square(df, target_column, variable_name):
    """
    Calculate the Chi-Square statistic and p-value for a variable against the target in a DataFrame for binary classification.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    target_column (str): Name of the binary target column (0 or 1).
    variable_name (str): Name of the variable for which to calculate the Chi-Square statistic.

    Returns:
    float: Chi-Square statistic
    float: p-value
    """
    contingency_table = pd.crosstab(df[target_column], df[variable_name])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p

def calculate_chi_square_values(df, target_column, alpha=0.05):
    """
    Calculate the Chi-Square statistic and p-values for all variables against the target in a DataFrame for binary classification.
    Add a column indicating significance (1 for significant, 0 for not significant).

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    target_column (str): Name of the binary target column (0 or 1).
    alpha (float): Significance level for p-values.

    Returns:
    pd.DataFrame: DataFrame containing the Chi-Square statistic, p-value, and a significance column for each variable.
    """
    chi_square_scores = {}

    for col in df.columns:
        if col != target_column:
            chi2, p = calculate_chi_square(df, target_column, col)
            significance = 1 if p < alpha else 0
            chi_square_scores[col] = chi2, p, significance

    chi_square_df = pd.DataFrame.from_dict(chi_square_scores, orient='index', columns=['Chi-Square', 'p-value', 'Significance'])

    return chi_square_df

# Example usage:
# Assuming your DataFrame is named 'train2' and you want to calculate Chi-Square statistics and p-values for all variables against the target
target_column = 'churn'  # Change this to the name of your binary target column
alpha = 0.05  # Significance level
chi_square_df = calculate_chi_square_values(df_train_2.drop(columns=['mobile_number','last_date_of_month'], axis=1), 'churn', alpha=0.05)

# Print or use the DataFrame
chi_square_df.head()

# %%
len(df_train_2.drop(columns=['mobile_number','last_date_of_month'], axis=1).columns)

# %%
chi_square_df[chi_square_df.Significance>0]

# %%
chi_square_df = chi_square_df.reset_index()
chi_square_df.head()

# %%
chi_square_df

# %%
chi_square_df['index'].unique()

# TODO ok, so what. Te faltan muchas explicaciones de por qué hiciste algo. A qué conclusión llegaste?

# %%
cols_model2 = ['arpu', 'onnet_mou', 'offnet_mou', 'roam_ic_mou', 'roam_og_mou',
       'loc_og_t2t_mou', 'loc_og_t2m_mou', 'loc_og_t2f_mou',
       'loc_og_t2c_mou', 'loc_og_mou', 'std_og_t2t_mou', 'std_og_t2m_mou',
       'std_og_t2f_mou', 'std_og_t2c_mou', 'std_og_mou', 'isd_og_mou',
       'spl_og_mou', 'og_others', 'total_og_mou', 'loc_ic_t2t_mou',
       'loc_ic_t2m_mou', 'loc_ic_t2f_mou', 'loc_ic_mou', 'std_ic_t2t_mou',
       'std_ic_t2m_mou', 'std_ic_t2f_mou', 'std_ic_t2o_mou', 'std_ic_mou',
       'total_ic_mou', 'spl_ic_mou', 'isd_ic_mou', 'ic_others',
       'total_rech_num', 'total_rech_amt', 'max_rech_amt',
       'last_day_rch_amt', 'total_rech_data', 'max_rech_data',
       'count_rech_2g', 'count_rech_3g', 'av_rech_amt_data', 'vol_2g_mb',
       'vol_3g_mb', 'arpu_3g', 'arpu_2g', 'night_pck_user', 'monthly_2g',
       'sachet_2g', 'monthly_3g', 'sachet_3g', 'fb_user',
       'day_of_last_rech', 'day_of_last_rech_data']

# %%
len(cols_model2)

# %%
# Create a DataFrame containing only the selected columns
df_train_2_selected = df_train_2[cols_model2]

y_train_2 = df_train_2['churn']

# Initialize a Random Forest classifier
rf_classifier_2 = RandomForestClassifier(random_state=42)

# Train the Random Forest classifier on the selected features
rf_classifier_2.fit(df_train_2_selected, y_train_2)

# Get feature importances from the trained model
feature_importances_2 = rf_classifier_2.feature_importances_

# Create a DataFrame to store feature importances
importance_df_2 = pd.DataFrame({'Feature': df_train_2_selected.columns, 'Importance': feature_importances_2})

# Sort the DataFrame by feature importance in descending order
importance_df_2 = importance_df_2.sort_values(by='Importance', ascending=False)

# Print or use the importance DataFrame
print(importance_df_2)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Assuming you have already created 'importance_df' as in your code

# Sort the DataFrame by feature importance in descending order
importance_df_2 = importance_df_2.sort_values(by='Importance', ascending=False)

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_2, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

# %%
columns_selected = [
    'arpu',
    'day_of_last_rech',
    'offnet_mou',
    'total_rech_num',
    'loc_ic_mou',
    'total_ic_mou',
    'total_rech_amt',
    'total_og_mou',
    'onnet_mou',
    'last_day_rch_amt',
    'loc_ic_t2m_mou',
    'max_rech_amt',
    'loc_og_mou',
    'roam_ic_mou',
    'loc_ic_t2t_mou',
    'loc_og_t2m_mou',
    'roam_og_mou',
    'loc_og_t2t_mou',
    'std_og_mou',
    'spl_og_mou',
    'std_ic_mou',
    'loc_ic_t2f_mou',
    'std_og_t2m_mou',
    'std_ic_t2m_mou',
    'std_og_t2t_mou',
    'std_ic_t2t_mou',
    'loc_og_t2f_mou',
    'day_of_last_rech_data',
    'loc_og_t2c_mou',
    'vol_3g_mb',
    'ic_others',
    'av_rech_amt_data',
    'vol_2g_mb',
    'og_others',
    'isd_ic_mou',
    'std_ic_t2f_mou',
    'spl_ic_mou',
    'max_rech_data',
    'total_rech_data',
    'arpu_3g',
    'arpu_2g',
    'std_og_t2f_mou',
    'fb_user'
]


# %%
len(columns_selected)

# %% [markdown]
# Correlaciones

# %% [markdown]
# no filters here

# %%
# Calculate the pairwise correlation matrix among the selected features
correlation_matrix = df_train_2[columns_selected+['churn']].corr()

# Identify features with correlation > 0.8 with any other feature
highly_correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            highly_correlated_pairs.append((colname_i, colname_j, correlation_matrix.iloc[i, j]))

# Print or use the highly correlated feature pairs and their correlation values
for pair in highly_correlated_pairs:
    print(f"Features '{pair[0]}' and '{pair[1]}' have correlation: {pair[2]:.2f}")

# %%
# Calculate the pairwise correlation matrix among the selected features
correlation_matrix = df_train_2[columns_selected + ['churn']].corr()

# Set the correlation threshold (e.g., 0.8)
correlation_threshold = 0.8

# Identify features with correlation > correlation_threshold with any other feature
highly_correlated_columns = set()  # Use a set to store unique column names

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            highly_correlated_columns.add(colname_i)
            highly_correlated_columns.add(colname_j)

# Convert the set of highly correlated columns to a list
highly_correlated_columns_list = list(highly_correlated_columns)

# Print or use the list of highly correlated columns
print("Highly correlated columns:", highly_correlated_columns_list)


# %%
final_features = columns_selected

# %% [markdown]
# ## Model Selected

# %%
df_train_2.shape

# %%
exp2 = setup(data=df_train_2[final_features+['churn']], target='churn', session_id=123,fix_imbalance=False)

# %%
best_model = compare_models()

# %%
lightgbm = create_model('lightgbm')

# %%
tuned_lightgbm = tune_model(lightgbm,optimize='f1')

# %%
plot_model(tuned_lightgbm, plot='feature')

# %%
hyperparameters = lightgbm.get_params()

# Print the hyperparameters
print("Tuned LightGBM Hyperparameters:")
print(hyperparameters)

# %% [markdown]
# ## Final Model 2 Cliente-mes

# %%
import lightgbm as lgb
X = df_train_2[final_features]
y = df_train_2['churn']


X_predict = df_oot_2[X.columns].copy()

# %%
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42,stratify= y)

# %%
train_data = lgb.Dataset(X_train , label=y_train)
test_data = lgb.Dataset(X_test , label=y_test)

# %%
train_data = lgb.Dataset(X_train , label=y_train)
test_data = lgb.Dataset(X_test , label=y_test)

# %%
parameters = {
    'boosting_type': 'gbdt',
          'max_depth' : 4,
          'objective': 'binary',
          'nthread': 6, 
          'num_leaves': 31,
          'learning_rate': 0.1,
          'max_bin': 512,
          'subsample_for_bin': 2000,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.85,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 0.001,
          'min_child_samples': 20,
          'scale_pos_weight': 1,
          'feature_fraction': 0.85,
          'bagging_fraction': 0.85,
          'num_class' : 1,
          'is_unbalance': 'true',
          'metric' : 'auc',
          'verbose': -1,
          
}

# %%
modelo_lgb = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=2000)

# %%
## Feature importance

# %%
feature_imp = pd.DataFrame(sorted(zip(modelo_lgb.feature_importance(),X_train.columns)), columns=['Importancia','Variable'])
feature_imp.sort_values(by='Importancia', ascending=False, inplace=True)   

# %%
plt.figure(figsize=(15,8))
sns.barplot(x="Importancia", y="Variable", data=feature_imp.head(20))

# %%
prediccion_lgb_test = modelo_lgb.predict(X_test, num_iteration=modelo_lgb.best_iteration)
prediccion_lgb_train = modelo_lgb.predict(X_train, num_iteration=modelo_lgb.best_iteration)

# %%
prediccion_lgb_submit = modelo_lgb.predict(X_predict, num_iteration=modelo_lgb.best_iteration)

# %%
prediccion_lgb_test.min(), prediccion_lgb_test.mean(), prediccion_lgb_test.max()

# %%
prediccion_lgb_submit.min(), prediccion_lgb_submit.mean(), prediccion_lgb_submit.max()

# %%
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Define a threshold (e.g., 0.5) 
threshold = 0.5

# Convert probabilities to binary predictions
predictions_test = (prediccion_lgb_test >= threshold).astype(int)
predictions_train = (prediccion_lgb_train >= threshold).astype(int)
predictions_submit = (prediccion_lgb_submit >= threshold).astype(int)

# Calculate F1-score
f1_score_test = f1_score(y_test, predictions_test)
f1_score_train = f1_score(y_train, predictions_train)
f1_score_oot = f1_score(df_oot_2['churn'], predictions_submit)

# Calculate confusion matrix
confusion_matrix_test = confusion_matrix(y_test, predictions_test)
confusion_matrix_train = confusion_matrix(y_train, predictions_train)
confusion_matrix_oot = confusion_matrix(df_oot_2['churn'], predictions_submit)

# Print or use the F1-scores and confusion matrices
print("F1-score (Test):", f1_score_test)
print("F1-score (Train):", f1_score_train)
print("F1-score (OOT):", f1_score_oot)
print("Confusion Matrix (Test):\n", confusion_matrix_test)
print("Confusion Matrix (Train):\n", confusion_matrix_train)
print("Confusion Matrix (oot):\n", confusion_matrix_oot)

# %%
df_oot_2.shape

# %%
1199+93750+683+3267

# %% [markdown]
# ### Explainer shap

# %%
import shap
import matplotlib.pyplot as plt

# Create a SHAP explainer for your LightGBM model
explainer = shap.TreeExplainer(modelo_lgb)

# Get SHAP values for the entire dataset
shap_values = explainer.shap_values(df_train_2[final_features])

# Plot SHAP summary plot
shap.summary_plot(shap_values, df_train_2[final_features])

# Show the plot
plt.show()

# %%


# %%
### Understanding the arpu

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named df_oot_2
df_temp = df_oot_2.copy()

# Create quintiles for the 'arpu' column
df_temp['arpu_quintile'] = pd.qcut(df_temp['arpu'], q=20, labels=False)

# Calculate the average of 'churn' for each quintile group
quintile_averages = df_temp.groupby('arpu_quintile')['churn'].mean()

# Create a line plot for the average churn in each quintile
plt.plot(quintile_averages, marker='o', color='blue', linestyle='-', linewidth=2)

# Set labels and title
plt.xlabel('ARPU Quintiles')
plt.ylabel('Average Churn')
plt.title('Average Churn in Each ARPU Quintile')

# Show the plot
plt.show()


# %% [markdown]
# # Anlysis Profile

# %%
df_oot_2.head()

# %%
df_oot_2['predictions_label'] = predictions_submit
df_oot_2['prediccion_prob'] = prediccion_lgb_submit

# %%
df_oot_2.head()

# %%
df_oot_2['prediccion_prob_groups'] = pd.qcut(df_oot_2['prediccion_prob'], q=3, labels=['Low_prob', 'Medium_prob', 'High_prob'])

# %%
df_oot_2['prediccion_prob_groups'].value_counts()

# %%
df_oot_2.groupby('prediccion_prob_groups')['churn'].mean()

# %%
day_of_last_rech
arpu
last_day_rch_amt
loc_og_t2t_mou
loc_og_t2m_mou

# %%
import pandas as pd

# Assuming you have a DataFrame named df_temp2_oot_2
df_temp2 = df_oot_2.copy()

# List of variables for which you want to create quintiles and calculate average churn
variables = ['day_of_last_rech', 'arpu', 'last_day_rch_amt', 'loc_og_t2t_mou', 'loc_og_t2m_mou']

# Loop through the list of variables
for variable in variables:
    # Create quintiles for the variable
    df_temp2['{}_quintile'.format(variable)] = pd.qcut(df_temp2[variable], q=5,duplicates='drop')

# %%
df_temp2.head()

# %%
df_temp2.groupby(['prediccion_prob_groups','arpu_quintile'])['mobile_number'].count().reset_index()

# %%
import pandas as pd

# Assuming you have a DataFrame named df_temp2
# Group by 'prediccion_prob_groups' and 'arpu_quintile' and calculate the count of 'mobile_number'
grouped_counts = df_temp2.groupby(['prediccion_prob_groups', 'day_of_last_rech_quintile'])['mobile_number'].count().reset_index()

# Calculate the total count of 'mobile_number' within each 'prediccion_prob_groups' group
total_counts = grouped_counts.groupby('prediccion_prob_groups')['mobile_number'].transform('sum')

# Calculate the percentage by dividing the count by the total count
grouped_counts['percentage'] = (grouped_counts['mobile_number'] / total_counts) * 100

# Add the name of the variable you're analyzing in a new column
grouped_counts['variable'] = 'day_of_last_rech_quintile'  # Change 'arpu_quintile' to the appropriate variable name

# Display the result
grouped_counts

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ## Return on Investment - ROI 

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# # Respondiendo la consulta de esta pregunta

# %%
## Consultas

# %% [markdown]
# a. ¿Qué métricas usaste para evaluar el desempeño del modelo? ¿Por qué?
# 
# b. ¿Cómo podrías mejorar la performance del modelo?
# 
# c. ¿Por qué elegiste este modelo

# %%
## Respuestas:

# %% [markdown]
# a. La metrica a usar sería el F1-score debido a la data desbalanceada.
# 
# b. Mejoran teniendo más historia para usar comportamiento histórico
# 
# c. Debido a que presenta mejor F1-score con respecto al resto de modelos

# %% [markdown]
# ## Conclusiones
# 
# 1. Generando el modelo con la metodologia cliente - mes, trae mejores resultados debido a la cantidad de observaciones
# 2. Las variables top 5 más importantes son: 
# - 'day_of_last_rech': dia desde su ultima recarga
# - 'arpu', average revenue per client
# - 'last_day_rch_amt', monto de su ultima recarga amt
# - 'loc_og_t2t_mou', local call outgoing to i.e. operator in minutes of use
# - 'loc_og_t2m_mou',local call outgoing to other operator in minutes of use
# 
# 4. El uso de un campaña con hip: con costo 1usd y beneficio 2usd, evidencia la ganancia del modelo en el negocio.
# 5. Se plantea realizar un a/b testing al grupo con mayor prob de churn para poder probar distintos incentivos

# %%


# %%



