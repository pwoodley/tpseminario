# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:12:27 2020

@author: Woodley
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
import joblib
from collections import OrderedDict
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing, metrics
import gc
import os
from itertools import cycle
import matplotlib as mpl

pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# Encontré esta función que reduce mucho lo que hay en memoria
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# funcion para leer la data y hacer un merge


def read_data():
    print('Reading files...')
    calendar = pd.read_csv(r'C:\Users\patri\Desktop\M5-Forecasting\calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(r'C:\Users\patri\Desktop\M5-Forecasting\sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(r'C:\Users\patri\Desktop\M5-Forecasting\sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(r'C:\Users\patri\Desktop\M5-Forecasting\sample_submission.csv')
    return calendar, sell_prices, sales_train_validation, submission


def melt_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows = 55000000, merge = False):
    
    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    sales_train_validation = reduce_mem_usage(sales_train_validation)
    
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]

    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
                      'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']

    # tabla de productos
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    # merge con la tabla de productos
    test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test1 = test1.merge(product, how = 'left', on = 'id')
    test2 = test2.merge(product, how = 'left', on = 'id')
    test2['id'] = test2['id'].str.replace('_validation','_evaluation')


    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    sales_train_validation['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'

    data = pd.concat([sales_train_validation, test1, test2], axis = 0)

    del sales_train_validation, test1, test2

    # En caso de que quiera solo usar una parte del dataset
    data = data.loc[nrows:]

    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    
    data = data[data['part'] != 'test2']
    
    if merge:
        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
        data.drop(['d', 'day'], inplace = True, axis = 1)
        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    else:
        pass

    gc.collect()

    return data

calendar, sell_prices, sales_train_validation, submission = read_data()
data = melt_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows = 0, merge = True)

# Realizo un poco de EDA antes de hacer Feature Engineering

# Grafico de ventas en el tiempo para un producto en particular
def plot_single_product(item_store_id):
    d_cols = [c for c in sales_train_validation.columns if 'd_' in c] # columnas de ventas

    sales_train_validation.loc[sales_train_validation['id'] == 'FOODS_3_120_CA_3'] \
        [d_cols[-365:]] \
        .T \
        .plot(figsize=(15, 5),
              title=item_store_id  + ' sales by "d" number',
              color=next(color_cycle))
    plt.legend('')
    plt.show()

plot_single_product('HOUSEHOLD_1_241_CA_1_validation')

plot_single_product('HOBBIES_1_021_CA_1_validation')

plot_single_product('FOODS_3_780_CA_1_validation')

# Cual es la categoria que mas vende?

cat = data.groupby('cat_id')['cat_id',"demand"].sum()

ax = sns.barplot(x=cat.index,y="demand", data=cat)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()

# Que tienda vende mas en cada estado?

storesales = data.groupby(['state_id','store_id'])['state_id','store_id','demand'].sum()
storesales = storesales.reset_index()

ax = sns.barplot(x='state_id',y="demand",hue='store_id', data=storesales)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()


#Top 10 productos que mas venden en todas las tiendas en ese periodo

prodsales = data.groupby(['item_id'])['item_id','demand'].sum()
prodsales = prodsales.reset_index()
prodsales.sort_values(by=['demand'],inplace=True,ascending=False)
prodsales = prodsales[0:9]
ax = sns.barplot(x='item_id',y="demand", data=prodsales)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()


# Realizo Feature Engenieering

def transform(data):

    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])

    return data

def simple_fe(data):

    # rolling demand features
    data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
    data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
    data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
    data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())


    # price features
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)

    # time features
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek


    return data

def run_lgb(data):

    # Se van a evaluar los ultimos 28 dias
    x_train = data[(data['date'] <= '2016-03-27')  & (data['date'] >= '2015-03-27')]
    y_train = x_train['demand']
    x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    y_val = x_val['demand']

    x_train = x_train[features]
    x_val = x_val[features]

    test = data[(data['date'] > '2016-04-24')]
    del data
    gc.collect()


    #LIGHT GBM
    # Classifier
    # Uso optimizacion bayesiana (100 iteraciones) para buscar los hiperparametros
    opt = BayesSearchCV(
        estimator = lgb.LGBMRegressor(
            n_jobs=-1,
            verbose=0,
        ),
        search_spaces = {
            'learning_rate': Real(1e-7, 1.0, 'log-uniform'),
            'num_leaves': Integer(2, 100),      
            'max_depth': Integer(0, 100),
            'min_child_samples': Integer(1, 50),
            'max_bin': Integer(100, 1000),
            'subsample': Real(0.01, 1.0, 'uniform'),
            'subsample_freq': Integer(0, 10),
            'colsample_bytree': Real(0.01, 1.0, 'uniform'),
            'min_child_weight': Real(1e-7, 10),
            'subsample_for_bin': Integer(100000, 500000),
            'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
            'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
            'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),
            'n_estimators': Integer(50, 500),
        },
        cv = TimeSeriesSplit(
            n_splits=3,
        ),
        n_jobs = -1,
        n_iter = 100,
        verbose = 0,
        scoring = 'neg_root_mean_squared_error',
        refit = True,
        random_state = 42,

    )

    # CALLBACKS para el optimizador bayesiano

    counter = 0

    checkpoint_saver = CheckpointSaver("./checkpoint.pkl", compress=9) # keyword arguments will be passed to `skopt.dump`

    def onstep(res):
        # global counter
        nonlocal counter
        args = res.x
        x0 = res.x_iters
        y0 = res.func_vals
        print('Last eval: ', x0[-1], 
            ' - Score ', y0[-1])
        print('Current iter: ', counter, 
            ' - Score ', res.fun, 
            ' - Args: ', args)
        joblib.dump((x0, y0), 'checkpoint.pkl')
        counter = counter+1


    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        all_models = pd.DataFrame(opt.cv_results_) 

        best_params = pd.Series(opt.best_params_)

        clf_name = opt.estimator.__class__.__name__

        print('Model #{}\nBest RMSE: {}\nBest params: {}\n'.format(len(all_models),np.round(-opt.best_score_, 4),opt.best_params_))


        all_models.to_csv(clf_name+"_cv_results.csv")

    # entreno el modelo y busco el mejor
    result = opt.fit(x_train, y_train, callback=[checkpoint_saver,onstep, status_print])

    return result

def predict(test, submission):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv('submission.csv', index = False)


# defino la lista de features
features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'dayofweek', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90', 
            'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']


def transform_train_and_eval(data):
    data = transform(data)
    data = simple_fe(data)
    data = reduce_mem_usage(data)
    print("Saving to csv")
    data.to_csv("final_dataset.csv")
    result = run_lgb(data)


transform_train_and_eval(data)

# Leo de vuelta el dataset para poder calcular tambien en validation y en test. (Podría haberlo hecho todo junto)

data_final = pd.read_csv("final_dataset.csv", header=0)
# submission = pd.read_csv(r'C:\Users\patri\Desktop\M5-Forecasting\sample_submission.csv')
data_final = reduce_mem_usage(data_final)

x_train = data_final[(data_final['date'] <= '2016-03-27')  & (data_final['date'] >= '2015-03-27')]
y_train = x_train['demand']
x_val = data_final[(data_final['date'] > '2016-03-27') & (data_final['date'] <= '2016-04-24')]
y_val = x_val['demand']

x_train = x_train[features]
x_val = x_val[features]

test = data_final[(data_final['date'] > '2016-04-24')]
del data_final
gc.collect()

# Leo el csv que se generó con la optimización bayesiana
lgbmresults = pd.read_csv("LGBMRegressor_cv_results.csv")

#Me quedo con los primeros 50 
lgbmresults = lgbmresults.loc[lgbmresults.rank_test_score <=50,:]

# Calculo el rmse para el dataset de validacion


def rmse_val(ordereddicc):
    global x_train, y_train, x_val, y_val
    modelo = eval('lgb.LGBMRegressor(**{b})'.format(b=ordereddicc))
    modelo.fit(x_train,y_train)

    val_pred = modelo.predict(x_val)

    val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))

    return val_score

# Calculo el rmse para validation para los 50 mejores modelos
lgbmresults['rmse_val'] = lgbmresults.params.apply(rmse_val)

lgbmresults.to_csv('LGBMRegressor_cv_results_mod.csv')

# Elijo el mejor modelo (Con buen RMSE tanto en train como en validation)
# Hago la predicción en test

bestparams = OrderedDict([('colsample_bytree', 0.3523974426705613), ('learning_rate', 0.015779225888570857), ('max_bin', 930), ('max_depth', 100), ('min_child_samples', 31), ('min_child_weight', 10.0), ('n_estimators', 500), ('num_leaves', 100), ('reg_alpha', 0.0001565772428821962), ('reg_lambda', 1e-09), ('scale_pos_weight', 2.3165605184104963), ('subsample', 0.7536523883731187), ('subsample_for_bin', 104108), ('subsample_freq', 3)])

bestmodel = eval('lgb.LGBMRegressor(**{b})'.format(b=bestparams))

bestmodel.fit(x_train,y_train)

val_pred = bestmodel.predict(test[features])

test['demand'] = val_pred

test.to_csv("test.csv")

predict(test, submission)
