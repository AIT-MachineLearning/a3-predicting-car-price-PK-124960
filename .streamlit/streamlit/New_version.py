import streamlit as st
import pickle
import numpy as np
# from sklearn.model_selection import KFold
from LinearClass import *
#from annotated_text import annotated_text

import os
base_path:str = os.getcwd()
filename = os.path.join(base_path, "trained_model_v2.1.sav")
loaded_model = pickle.load(open(filename, 'rb'))


def car_predict(name, engine, max_power, mileage):
    input = np.array([[name, engine, max_power, mileage, 1]]).astype(np.float64)
    prediction = loaded_model.predict(input) # string type
    predicted_price = float(prediction) * 1000
    pred = "{:,.2f}".format(predicted_price)
    return pred

def main():

    html_temp = """
    <div style="background-color:#2E8B57 ;padding:12px">
    <h2 style="color:white;text-align:center;">New Car Price Prediction ML APP </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.subheader("1. What kind of car are you selling?")
    option = st.selectbox(
        " ",
        ('Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
    'Mahindra', 'Tata', 'Chevrolet', 'Fiat', 'Datsun', 'Jeep',
    'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW',
    'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
    'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel', 'Peugeot'),
    )

    if option == 'Maruti':
        name = 20
    elif option == 'Skoda':
        name = 27
    elif option == 'Honda':
        name = 10
    elif option == 'Hyundai':
        name = 11
    elif option == 'Toyota':
        name = 29
    elif option == 'Ford':
        name = 9
    elif option == 'Renault':
        name = 26
    elif option == 'Mahindra':
        name = 19
    elif option == 'Tata':
        name = 28
    elif option == 'Chevrolet':
        name = 4
    elif option == 'Fiat':
        name = 7
    elif option == 'Datsun':
        name = 6
    elif option == 'Jeep':
        name = 14
    elif option == 'Mercedes-Benz':
        name = 21
    elif option == 'Mitsubishi':
        name = 22
    elif option == 'Audi':
        name = 2
    elif option == 'Volkswagen':
        name = 30
    elif option == 'BMW':
        name = 3
    elif option == 'Nissan':
        name = 23
    elif option == 'Lexus':
        name = 17
    elif option == 'Jaguar':
        name = 13
    elif option == 'Land':
        name = 16
    elif option == 'MG':
        name = 18
    elif option == 'Volvo':
        name = 31
    elif option == 'Daewoo':
        name = 5
    elif option == 'Kia':
        name = 15
    elif option == 'Force':
        name = 8
    elif option == 'Ambassador':
        name = 0
    elif option == 'Ashok':
        name = 1
    elif option == 'Isuzu':
        name = 12
    elif option == 'Opel':
        name = 24
    else:
        name = 25

    st.write("You selected:", (option, name))
    #name = st.text_input("Brand","Type Here")
    #st.write("Your Engine:")

    st.subheader("2. Car Engine")
    engine = st.slider("Select CC of car engine", 0, 4000)
    st.write("Your Car Engine in CC:", engine)

    st.subheader("3. The Maximum Power of The car")
    max_power = st.slider("Select bhp of horsepower", 0, 1000)
    st.write("Your Max Power in bhp:", max_power)

    st.subheader("4. Mileage")
    mileage = st.slider("Select kmpl of mileage", 0, 100)
    st.write("Your Mileage in kmpl:", mileage)

    # max_power = st.text_input("Max Power","Type Here")
    # mileage = st.text_input("Mile Age","Type Here")

    # safe_html = """  
    #    <div style="background-color:#C70039 ;padding:12px >
    #     <h2 style="color:white;text-align:center;">Created by Ponkrit Kaewsawee, st124960, DSAI</h2>
    #     </div>
    # """
    # danger_html="""  
    #   <div style="background-color:#F08080;padding:10px >
    #    <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
    #    </div>
    # """

    if st.button("Predict"):
        output = car_predict(name, engine, max_power, mileage)
        st.success('The Projected car selling price is {} bath'.format(output), icon="✅")
        #st.markdown(safe_html,unsafe_allow_html=True)

    #     annotated_text(
    #         "This project ",
    #         (" is "),
    #         (" created "),
    #         " by ",
    #         ("Mr.Ponkrit", "Kaewsawee"),
    #         ("st124960", "DSAI"),
    #         " at ",
    #         ("AIT.", "SET"),
    #         "."
    #     )
        # else:
        #     st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
