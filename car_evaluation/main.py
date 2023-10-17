import ML_model
import pickle

from flask import Flask, request, jsonify
import pandas as pd
import pickle


base_model,encoder, X_train = ML_model.ML_model()

# Serialize the model
with open("rf_model.pkl", "wb") as model_file:
    pickle.dump(base_model, model_file)

# Serialize the encoder (for one-hot encoding)
with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

# Serialize training columns for consistent input during prediction
with open("train_columns.pkl", "wb") as tc_file:
    pickle.dump(X_train.columns.tolist(), tc_file)

#this is the API part

app = Flask(__name__)

# Load the pickled model
with open("rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the pickled encoder
with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    COLOR_CODE = data["COLOR_CODE"]
    model_year = data["MODEL_YEAR"]
    car_age = data["car_age"]
    CAR_KIND_CODE = data["CAR_KIND_CODE"]
    CAR_METER = data["CAR_METER"]
    car_origin = data['car_origin']

    # # specify the categorical data that we need to set to 1
    # kind_str = 'kind_code_{}'.format(kind_code)
    # D_COLOR_str = 'D_COLOR_{}'.format(D_COLOR)
    #
    # # Create a dataframe to hold the input data
    # input_data = pd.DataFrame([[model_year, car_age, millage, 1, 1]],
    #                           columns=['model', 'car_age', 'meter_reading', kind_str, D_COLOR_str])


    #specify the categorical data that we need to set to 1
    CAR_KIND_CODE_str = 'CAR_KIND_CODE_{}'.format(CAR_KIND_CODE)
    COLOR_CODE_str = 'COLOR_CODE_{}'.format(COLOR_CODE)
    car_origin_str = 'car_origin_{}'.format(car_origin)



    # Create a dataframe to hold the input data, needs to be the same order as train set, the ones are there to set the specified featueres
    input_data = pd.DataFrame([[model_year,car_age,CAR_METER,1,1,1]],
                              columns=['MODEL_YEAR','car_age','CAR_METER',CAR_KIND_CODE_str,COLOR_CODE_str,car_origin_str])

    # Reindex as before
    with open("train_columns.pkl", "rb") as tc_file:
        train_columns = pickle.load(tc_file)
    input_data = input_data.reindex(columns=train_columns, fill_value=0)

    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)

    app.run(debug=True)
