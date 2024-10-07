import streamlit as st
import pickle
import numpy as np
# from sklearn.model_selection import KFold
# from LinearClassA3 import MultinomialLogisticRegression, RidgePenalty
# from LinearClassA3 import *
#from annotated_text import annotated_text
# class Ridge(MultinomialLogisticRegression):
#     def __init__(self, learning_rate, l, epochs=1000):
#         regularization = RidgePenalty(l)
#         super().__init__(regularization, learning_rate, epochs)

import os
base_path:str = os.getcwd()
filename1 = os.path.join(base_path, "trained_model_v3.sav")
loaded_model = pickle.load(open(filename1, 'rb'))
filename = os.path.join(base_path, "scaler.pkl")
scaler = pickle.load(open(filename, 'rb'))


def car_predict(km_driven, owner, mileage, max_power, engine):
    input = np.array([[km_driven, owner, mileage, max_power, engine]]).astype(np.float64)
    input = scaler.transform(input)
    prediction = loaded_model.predict(input) # string type
    # predicted_price = float(prediction)
    # pred = "{:,.2f}".format(predicted_price)
    return prediction

def main():

    html_temp = """
    <div style="background-color:#2E8B57 ;padding:12px">
    <h2 style="color:white;text-align:center;">New Car Price Prediction ML APP </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.subheader("1. KM driven")
    km_driven = st.slider("Select KM of Car dirven", 0, 2400000)
    st.write("Your Distance in KM:", km_driven)

    st.subheader("2. owner")
    owner = st.slider("Select Owner number", 0, 5)
    st.write("Your Owner number:", owner)

    st.subheader("3. The mileage of The car")
    mileage = st.slider("Select mileage", 0, 42)
    st.write("Your mileage:", mileage)

    st.subheader("4. max_power")
    max_power = st.slider("Select max_power", 0, 400)
    st.write("Your max_power:", max_power)

    st.subheader("5. engine")
    engine = st.slider("Select engine", 0, 3700)
    st.write("Your Engine:", engine)


    if st.button("Predict"):
        output = car_predict(km_driven, owner, mileage, max_power, engine)
        st.success('The Class Car is {}'.format(output), icon="✅")

if __name__=='__main__':
    main()
