from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import time

def ML_model():
    start_time = time.time()
    car_data = pd.read_csv("/Users/abdulrahmanalamar/Desktop/evaluation cars/Evaluation_Cars2.csv")
    count_cars = car_data['kind_desc'].value_counts()
    print(count_cars.to_markdown())
    # print(car_data.head(0))
    #drop the unwanted descriptive columns
    car_data = car_data.drop(["kind_desc","color_desc"], axis = 'columns')
    # print(car_data.head(0))

    #filter the data for frequency of kind code higher than 18
    car_data_filtered = car_data[car_data.groupby('kind_code')['kind_code'].transform('count')>=100]

    #show the count of cars after cleansing
    count_cars = car_data_filtered['kind_code'].value_counts()
    # print(count_cars.to_markdown())

    #calculate the correlation matrix
    corr_car_data = car_data_filtered.corr().round(2)
    # print(corr_car_data.to_markdown())

    #do one hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore',drop='first')
    encoded_columns = ['kind_code','D_COLOR']
    encoder_car_data_filtered = pd.DataFrame(encoder.fit_transform(car_data_filtered[encoded_columns]).toarray(),columns=encoder.get_feature_names_out(encoded_columns),index=car_data_filtered.index)

    #join encoded data to original
    final_car_data = car_data_filtered.join(encoder_car_data_filtered)
    final_car_data.drop(encoded_columns, axis=1, inplace=True)  # drop original kind_code column
    # print(final_car_data.to_markdown())

    # data preparation
    feature_columns = ['model','car_age','meter_reading']+list(encoder.get_feature_names_out())[1:]
    print(feature_columns)
    X = final_car_data[feature_columns]
    Y = final_car_data['initial_price']

    #Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.6, random_state=1)

    #instantiate the model for lienar regression
    model = LinearRegression()
    #fit the model
    model.fit(X_train,Y_train)
    # get the prediction
    predicted_Y_test = model.predict(X_test)
    # get the error for the prediction
    print("\nThis is the regular linear regression data mean squared error: " + str(np.sqrt(mean_squared_error(Y_test, predicted_Y_test)).round(2)))

    # print(np.sqrt(mean_squared_error(Y_test,predicted_Y_test).round(2)))

    # this is an exhastive test for figuring out the best params for the random forest model

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

    # Instantiate the grid search model

    # Evaluate the models
    base_model = RandomForestRegressor(n_estimators=1600, random_state=42)
    base_model.fit(X_train, Y_train)
    base_accuracy = evaluate(base_model, X_test, Y_test)
    linear_regression_accuracy = evaluate(model,X_test,Y_test)


    print(f"Base Model Accuracy: {base_accuracy:.2f}%")
    print(f"Linear regression model Accuracy: {linear_regression_accuracy:.2f}%")



    #======================================={ Testing human input and prediction }====================

    # D_COLOR = int(input("Enter D_COLOR value: "))
    # model_year = int(input("Enter model year: "))
    # car_age = int(input("Enter car age: "))
    # kind_code = input("Enter kind code: ")
    # millage = input("Enter Millage: ")

    #hard coded inputs for testing purposes
    D_COLOR = 2004.0
    model_year = 2016
    car_age = 3
    kind_code = 3642.0
    millage = 50000

    #specify the categorical data that we need to set to 1
    kind_str = 'kind_code_{}'
    D_COLOR_str = 'D_COLOR_{}'
    D_COLOR_str = D_COLOR_str.format(D_COLOR)
    kind_str = kind_str.format(kind_code)

    # Create a dataframe to hold the input data, needs to be the same order as train set, the ones are there to set the specified featueres
    input_data = pd.DataFrame([[model_year,car_age,millage,1,1]],
                              columns=['model','car_age','meter_reading',kind_str,D_COLOR_str])

    # since we're using one-hot-encoding, we need to set the binary feature columns to ON and zero out the rest of the columns. hence the fill value=0
    input_data = input_data.reindex(columns=X_train.columns,fill_value=0)

    manual_input_prediction = base_model.predict(input_data)
    print(manual_input_prediction)

    #calculating time elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    #work on derived features
    #maybe look into the decay per car or per origin manufacturer per year and quantify that and add it as a feature.
    #try to guess the profit margin using ensemble training by feeding the prediction into the linear regression module and so forth
    return base_model, encoder, X_train
