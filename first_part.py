# %% [markdown]
# ### Librerias

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ### Funciones

# %%
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

# %% [markdown]
# ### Cargamos la base

# %%
path = ''
df = pd.read_csv(path+'./dataset_prueba.csv')

# %%
df.head()

# %% [markdown]
# ### EDA

# %%
df.shape

# %%
df.info()

# %%
### Datos del enunciado
target = 'churn'
IDS = ['mobile_number','last_date_of_month']

# %%
### obs de los ids
df[IDS]

# %%
## reviando duplicados
df['mobile_number'].nunique(), len(df)

# %%
df[df.mobile_number==7000000074][['last_date_of_month','churn','arpu']]

# %%
## revisando data historica
df['last_date_of_month'].value_counts(dropna=False) ## missing values in ID column # TODO Tienes un error acá que tuve que corregir. Siempre verifica que tu código funcione de principio a fin antes de entregarlo

# %%
## revisando duplicados
agr = {'mobile_number':['count','nunique']}
result = df.groupby('last_date_of_month').agg(agr).reset_index()
result.columns = ['last_date_of_month', 'mobile_number_count', 'mobile_number_nunique']
result

# %%
# revisando nulos
df.isnull().sum()

# %% [markdown]
# ### 1. First question

# %% [markdown]
# ## 1.1 New column named clasificacion_clientes_revenu

# %%
df['deciles'], bins = pd.qcut(df['arpu'], q=10, retbins=True,labels = False)

# %%
df['deciles'].value_counts(dropna=False)

# %%
df['clasificacion_clientes_revenue'] = np.where(df['deciles']==9,'platino',
                                               np.where(df['deciles']==8,'gold','normal'))

# %%
#### Respuesta
df['clasificacion_clientes_revenue'].value_counts(dropna=False)

# %% [markdown]
# ###### Nota: tomando en cuenta que se están usando todos los registros y los clientes pueden aparecer más de una vez 

# %% [markdown]
# ## 1.2 Crea una columna binaria con nombre flag_recarga

# %%
df['flag_recarga']=np.where(df['total_rech_num']>0,1,0)
df['flag_recarga'].value_counts(dropna=False)

# %% [markdown]
# ## 1.3. ¿Cuál es la proporción (entre 0 y 1) de churn de cada mes, junio, julio y agosto?

# %%
df['churn'].value_counts()

# %%
df['last_date_of_month'] = pd.to_datetime(df['last_date_of_month'])

agr2 = {'churn':['sum','count']}
result3 = df.groupby('last_date_of_month').agg(agr2).reset_index()
result3.columns = ['last_date_of_month','churn_sum','churn_count']

# %%
result3['%_0/1'] = (result3['churn_count']-result3['churn_sum'])/(result3['churn_sum']) 
result3.head()

df.groupby("last_date_of_month")['churn'].mean()

# %% [markdown]
# ###### Nota: En el mes de junio hay 95 ceros por cada uno, se esta enviando la cantidad de 0 sobre la cantidad de 1

# %% [markdown]
# ## 1.4.Elimina las columnas que contienen sobre 70% de valores nulos, adjuntar número de
# ## columnas restantes. ¿Hay columnas que no deberían tener valores nulos?

# %%
null_prc = (df.isnull().sum() / len(df)) * 100
null_prc

# %%
columns_to_drop = null_prc[null_prc > 70].index

# %%
df2 = df.drop(columns=columns_to_drop)

# %%
### se deben borrar 10 c0lumnas
len(columns_to_drop)

# %% [markdown]
# ¿Hay columnas que no deberían tener valores nulos?: la columna last date of month no deberia estar en missing

# %%
df2.shape

# %%
cols = ['last_date_of_month', 'arpu', 'onnet_mou', 'offnet_mou', 'roam_ic_mou',
       'roam_og_mou', 'loc_og_t2t_mou', 'loc_og_t2m_mou', 'loc_og_t2f_mou',
       'loc_og_t2c_mou', 'loc_og_mou', 'std_og_t2t_mou', 'std_og_t2m_mou',
       'std_og_t2f_mou', 'std_og_t2c_mou', 'std_og_mou', 'isd_og_mou',
       'spl_og_mou', 'og_others', 'total_og_mou', 'loc_ic_t2t_mou',
       'loc_ic_t2m_mou', 'loc_ic_t2f_mou', 'loc_ic_mou', 'std_ic_t2t_mou',
       'std_ic_t2m_mou', 'std_ic_t2f_mou', 'std_ic_t2o_mou', 'std_ic_mou',
       'total_ic_mou', 'spl_ic_mou', 'isd_ic_mou', 'ic_others',
       'total_rech_num', 'total_rech_amt', 'max_rech_amt', 'date_of_last_rech',
       'last_day_rch_amt', 'date_of_last_rech_data', 'total_rech_data',
       'max_rech_data', 'count_rech_2g', 'count_rech_3g', 'av_rech_amt_data',
       'vol_2g_mb', 'vol_3g_mb', 'arpu_3g', 'arpu_2g', 'night_pck_user',
       'monthly_2g', 'sachet_2g', 'monthly_3g', 'sachet_3g', 'fb_user',
       'churn', 'mobile_number']

# %%
#1.4.2

# %%
temporal = df[cols]
summary = custom_describe(temporal)
summary

# %% [markdown]
# Estas columnas las borraría std_og_t2c_mou, std_ic_t2o_mou tienen una desviacion igual a 0 por tanto no aportan valor

# %% [markdown]
# La columna last_date_of_month, no deberia tener nulos ya que corresponde a la carga de datos.

# %% [markdown]
# ## 1.5.  Si un cliente hace churn o no, ¿se observan diferencias en la distribución de la variable
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Supongamos que tienes un DataFrame 'df' con las columnas 'total_rech_num' y 'churn'.

# Filtra los datos para separar a los clientes que hacen churn de los que no hacen churn
churned = df[df['churn'] == 1]
not_churned = df[df['churn'] == 0]

# Crea un gráfico de densidad para comparar las distribuciones
plt.figure(figsize=(10, 6))
sns.kdeplot(churned['total_rech_num'], label='Churned', color='red')
sns.kdeplot(not_churned['total_rech_num'], label='Not Churned', color='blue')
plt.title('Distribución de total_rech_num por Churn')
plt.xlabel('total_rech_num')
plt.ylabel('Densidad')
plt.legend()
plt.show()

# %% [markdown]
# se observa que los clientes que NO REALIZA CHURN presentan mayor magnitud del numero total de recargas, 
# distribucion más a la derecha y también se observa outliers en ambos casos

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame 'df' con las columnas 'total_rech_num' y 'churn'.

# Filtra los datos para separar a los clientes que hacen churn de los que no hacen churn
churned = df[df['churn'] == 1]
not_churned = df[df['churn'] == 0] # TODO Esto ya lo hiciste, hacerlo de nuevo es redundante e induce a errores

# Crea un boxplot para comparar las distribuciones
plt.figure(figsize=(10, 6))
plt.boxplot([churned['total_rech_num'], not_churned['total_rech_num']], labels=['Churned', 'Not Churned'])
plt.title('Distribución de total_rech_num por Churn')
plt.ylabel('total_rech_num')
plt.show()

# %% [markdown]
# se presenta outliers de la variable total_Rech_num en ambos tipos de clientes (churn & not churn)

# %% [markdown]
# 

# %%
import lightgbm as lgb

print(lgb.__version__)


# %%
import optuna
print(optuna.__version__)


# %%
import pycaret
print(pycaret.__version__) # TODO error de copy paste


# %%



