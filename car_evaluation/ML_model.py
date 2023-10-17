from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb #light gradient boosting machine
import time
from sklearn.metrics import r2_score


###########################{ Data Cleansing }####################################

#calculates the car age based on the selling date and the model year
def calculate_car_age(df):
    """Calculate the age of the car for the entire dataframe."""
    # Extract the selling year from 'D_SELLING_DATE'
    df['selling_year'] = df['D_SELLING_DATE'].astype(str).str[:4]

    # Ensure that both selling_year and MODEL_YEAR are numeric types
    df['selling_year'] = pd.to_numeric(df['selling_year'], errors='coerce')
    df['MODEL_YEAR'] = pd.to_numeric(df['MODEL_YEAR'], errors='coerce')

    # Compute car age
    df['car_age'] = df['selling_year'] - df['MODEL_YEAR']

    # Replace NaN values with 0
    df['car_age'].fillna(0, inplace=True)
    df['car_age'] = df['car_age'].astype(int)
    return df


#unify the colors to single whole numbers without interior colours to simplify the data
def colour_description(df, column_name="COLOR_DESC"):
    df["car_colour"] = df[column_name].str.split('/').str.get(0)
    # Strip any leading or trailing spaces
    df["car_colour"] = df["car_colour"].str.strip()
    #fix color data redundancy by unifying the names
    df["car_colour"] = df["car_colour"].replace("برونزى","برونزي")
    df["car_colour"] = df["car_colour"].replace("بنى","بني")
    df["car_colour"] = df["car_colour"].replace("رمادى غ","رمادى غامق")
    df["car_colour"] = df["car_colour"].replace("رمادى","رمادي")
    df["car_colour"] = df["car_colour"].replace("عنابىغ","عنابي")
    df["car_colour"] = df["car_colour"].replace("عنابىغ","عنابي")
    df["car_colour"] = df["car_colour"].replace("فضى","فضي")
    df["car_colour"] = df["car_colour"].replace("صدفي","ابيض صدفي")
    return df
# fix the spelling erorrs in the columns to cluster data and unify data
def Car_description(df):
    #fix some car brand names
    df["company name"] = df["company name"].replace("لاندكروزر","تويوتا")
    df["company name"] = df["company name"].replace("باترول","نيسان")
    df["company name"] = df["company name"].replace("باترولPT-LE","نيسان")
    df["company name"] = df["company name"].replace("باترولPT-SE","نيسان")
    df["company name"] = df["company name"].replace("مرسيدس","مرسيدس")
    df["company name"] = df["company name"].replace("سوزوكى","سوزوكي")
    df["company name"] = df["company name"].replace("ميتسوبيشى","ميتسوبيشي")
    df["company name"] = df["company name"].replace("ميزراتى","ميزاراتى")
    df["company name"] = df["company name"].replace("هيونداى","هيونداي")
    #fix some car descriptions, but need to change the car kind code correspondingly

    return df

#change the color codes corresponding to each trimmed color
def color_grouping(df):
    df['COLOR_CODE'] = df.groupby('car_colour')['COLOR_CODE'].transform('min')
    return df
#return only used cars, and company names that aren't empty this can be made in SQL
def car_class(df, class_value):
    return df[(df['CAR_CLASS']==class_value) & (pd.notna(df['company name']))]

def car_company(df, class_value):
    return df[df['car_origin'] == class_value]

#add the car's origin to have a more focused model. feature derivation
def classify_origin(df, column_name="company name"):

    Japanese = ['Toyota', 'Honda', 'Nissan', 'Mazda', 'Subaru', 'Suzuki', 'Mitsubishi','نيسان','لكزس','ميتسوبيشي','تويوتا']
    American = ['Ford', 'Cheverolet','GMC','Dodge','Jeep','Cadilac','شفر']
    Korean = ['Kia','Hyundai','كيا']
    Chineese = []
    German = ['BMW','Mercedes','Audi']
    European = []

    df['car_origin'] = 'other company'

    df.loc[df[column_name].isin(Japanese), 'car_origin'] = 'Japan'
    df.loc[df[column_name].isin(American), 'car_origin'] = 'America'
    df.loc[df[column_name].isin(Korean), 'car_origin'] = 'Korea'
    return df

# this rounds the prices of the cars to the nearest hundredth
def rounding_prices(df):
    df['rounded_D_COST_PRICE'] = (df['D_COST_PRICE'] / 100).round() * 100
    return df

###########################{ Model Evaluation }####################################
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)

    non_zero_indices = test_labels != 0
    mape = 100 * np.mean(errors[non_zero_indices] / test_labels[non_zero_indices])

    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def compute_r_squared(actual, predicted):
    return r2_score(actual, predicted)

def compute_abolute_error(actual, predicted):
    return mean_absolute_error(actual,predicted)

###########################{ Hyperparameter Tuning }####################################
def hyperparameter_tuning(X_train, Y_train):
    param_grid = {
    'n_estimators': [1000, 1200, 1500,1700,1800,2000],
    'max_features': ['auto', 'sqrt', 'log2',0.6,0.7,0.8],
    #'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4,5, 6,7,8,9,10],
    'bootstrap': [True]
    }

    rf = RandomForestRegressor(random_state = 42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    print(best_params)
    return grid_search.best_estimator_

def tune_lgbm_hyperparameters(X_train, y_train):
    # Define the model
    model = lgb.LGBMRegressor()

    # Define the hyperparameters and their possible values
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [20, 40, 60, 80, 100],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Define grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    print(grid_result.best_params_)

    return grid_result.best_estimator_, grid_result.best_params_

###########################{ ML Model }####################################
def ML_model():
    car_data = pd.read_csv("/Users/abdulrahmanalamar/Desktop/evaluation cars/Evaluation Cars.csv")

    car_data = calculate_car_age(car_data)
    car_data = colour_description(car_data)
    car_data = color_grouping(car_data)
    car_data= Car_description(car_data)
    car_data = car_class(car_data, 'USED')
    car_data = classify_origin(car_data)


    #car_data = rounding_prices(car_data)
    # car_data = car_company(car_data,'Japan')


    # print((car_data.head(0)).to_markdown())

    #filter the data for frequency of kind code higher than 18
    car_data_filtered = car_data[car_data.groupby('CAR_KIND_CODE')['CAR_KIND_CODE'].transform('count')>=100]
    car_data_filtered = car_data[car_data.groupby('CAR_METER')['CAR_METER'].transform('min')>0]

    #drop the unwanted columns
    car_data_filtered = car_data_filtered.drop(["D_INVOICE_NO","D_SELLING_DATE","D_RETAIL_PRICE","D_BUY_PRICE","D_EQUIP_CHRG","D_TOTAL_DEBIT","D_INITIAL_PRICE","DOWN_PAYMENT","D_INST_AMOUNT","D_INSTS_QTY","D_INSTS_MONTHS","CAR_SECTION","COLOR_DESC"], axis = 'columns')

    car_data_filtered.to_csv('output_after_cleaning.csv', index=False, encoding='utf-8-sig')

    #show the count of cars after cleansing
    count_cars = car_data_filtered['CAR_KIND_DESC'].value_counts()
    # print(count_cars.to_markdown())
    # print(pd.Series(car_data_filtered['company name'].unique()).to_markdown())

    #calculate the correlation matrix
    numeric_df = car_data_filtered.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr().round(2)
    print(correlation_matrix.to_markdown())

    #do one hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore',drop='first')
    encoded_columns = ['CAR_KIND_CODE','COLOR_CODE','car_origin']
    encoder_car_data_filtered = pd.DataFrame(encoder.fit_transform(car_data_filtered[encoded_columns]).toarray(),columns=encoder.get_feature_names_out(encoded_columns),index=car_data_filtered.index)

    #join encoded data to original
    final_car_data = car_data_filtered.join(encoder_car_data_filtered)
    final_car_data.drop(encoded_columns, axis=1, inplace=True)  # drop original kind_code column

    final_car_data.to_csv('output_after_encoding.csv', index=False, encoding='utf-8-sig')
    # data preparation
    feature_columns = ['MODEL_YEAR','car_age','CAR_METER']+list(encoder.get_feature_names_out())[1:]
    #print(feature_columns)
    X = final_car_data[feature_columns]
    Y = final_car_data['D_COST_PRICE']

    #Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)


    #Intantiate the random forest regressor model
    base_model = RandomForestRegressor(bootstrap= True,max_features= 0.6, min_samples_leaf= 2, n_estimators= 1700, random_state=42)
    base_model.fit(X_train, Y_train)
    # get the prediction
    predicted_Y_test = base_model.predict(X_test)

    #Instantiate the lightgbm model
    base_model2 = lgb.LGBMRegressor(random_state=42,colsample_bytree= 0.8, learning_rate= 0.05, min_child_samples=10, n_estimators=100, num_leaves=31, subsample=0.8)
    base_model2.fit(X_train,Y_train)
    # get the prediction
    predicted_Y_test2 = base_model2.predict(X_test)


    #evaluate the model
    # print("\nRandom Forrest model")
    # evaluate(base_model, X_test, Y_test)
    # # get the error for the prediction
    # # print(f"Base Model RMSE: {compute_rmse(Y_test, predicted_Y_test):.2f}")
    # print(f"Base Model R^{2}: {compute_r_squared(Y_test, predicted_Y_test):.2f}")
    # print(f"Base Model mean absolute error: {compute_abolute_error(Y_test, predicted_Y_test):.2f}")
    #
    # print("\nLGB model")
    # evaluate(base_model2, X_test, Y_test)
    # # get the error for the prediction
    # # print(f"Base Model RMSE: {compute_rmse(Y_test, predicted_Y_test2):.2f}")
    # print(f"Base Model R^{2}: {compute_r_squared(Y_test, predicted_Y_test2):.2f}")
    # print(f"Base Model2 mean absolute error: {compute_abolute_error(Y_test, predicted_Y_test2):.2f}")



    #hyperparameter tuning

    # best_rf = hyperparameter_tuning(X_train, Y_train)
    # base_accuracy2 = evaluate(best_rf, X_test, Y_test)
    # print(f"Base Model2 Accuracy: {base_accuracy2:.2f}%")

    # best_model,best_hyperparameters = tune_lgbm_hyperparameters(X_train, Y_train)
    # predicted_Y_test2 = best_model.predict(X_test)
    # evaluate(best_model, X_test, Y_test)
    # print(f"Best Hyperparameters: {best_hyperparameters}")


    # #======================================={ Testing human input and prediction }====================
    #
    # # D_COLOR = int(input("Enter D_COLOR value: "))
    # # model_year = int(input("Enter model year: "))
    # # car_age = int(input("Enter car age: "))
    # # kind_code = input("Enter kind code: ")
    # # millage = input("Enter Millage: ")
    #
    #hard coded inputs for testing purposes
    COLOR_CODE = 404 #maybe add a lookup for this
    model_year = 2016
    car_age = 3
    CAR_KIND_CODE = 3642
    millage = 50000
    car_origin = 'Japan'

    #specify the categorical data that we need to set to 1
    CAR_KIND_CODE_str = 'CAR_KIND_CODE_{}'
    COLOR_CODE_str = 'COLOR_CODE_{}'
    car_origin_str = 'car_origin_{}'


    COLOR_CODE_str = COLOR_CODE_str.format(COLOR_CODE)
    CAR_KIND_CODE_str = CAR_KIND_CODE_str.format(CAR_KIND_CODE)
    car_origin_str = car_origin_str.format(car_origin)

    # Create a dataframe to hold the input data, needs to be the same order as train set, the ones are there to set the specified featueres
    input_data = pd.DataFrame([[model_year,car_age,millage,1,1,1]],
                              columns=['MODEL_YEAR','car_age','CAR_METER',CAR_KIND_CODE_str,COLOR_CODE_str,car_origin_str])

    # since we're using one-hot-encoding, we need to set the binary feature columns to ON and zero out the rest of the columns. hence the fill value=0
    input_data = input_data.reindex(columns=X_train.columns,fill_value=0)
    print(input_data.to_markdown())

    manual_input_prediction = base_model.predict(input_data)
    manual_input_prediction_2 = base_model2.predict(input_data)
    print("\nthis is RF:")
    print(manual_input_prediction)
    print("\nThis is LGBM:")
    print(manual_input_prediction_2)


    # #work on derived features
    # #maybe look into the decay per car or per origin manufacturer per year and quantify that and add it as a feature.
    # #try to guess the profit margin using ensemble training by feeding the prediction into the linear regression module and so forth
    return base_model, encoder, X_train

start_time = time.time()
ML_model()

# calculating time elapsed
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time:.2f} minutes")

# {'bootstrap': True, 'max_features': 0.6, 'min_samples_leaf': 2, 'n_estimators': 1500} %85.82
# {'bootstrap': True, 'max_features': 0.6, 'min_samples_leaf': 2, 'n_estimators': 1700} %83.83

# {'bootstrap': True, 'max_features': 0.6, 'min_samples_leaf': 2, 'n_estimators': 1000}
# Base Model2 Accuracy: 86.08%

#{'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_samples': 10, 'n_estimators': 100, 'num_leaves': 31, 'subsample': 0.8}