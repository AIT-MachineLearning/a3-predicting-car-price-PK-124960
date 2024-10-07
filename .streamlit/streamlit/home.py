import streamlit as st

def app():

    html_temp = """
    <div style="background-color:#6B8E23 ;padding:12px">
    <h2 style="color:white;text-align:center;">Car Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.title('Welcome to :violet[ML @AIT.SET] :sunglasses:')

    st.markdown("## Need to sell your car for the best possible price? \n ")

    html_message1 = """
    <p style='font-family: Arial; font-size: 20px;'
    >Look no further! Our car price prediction tool, developed by AIT master student Mr.Ponkrit st.124960, can help you get the most out of your vehicle</p>
    """
    st.markdown(html_message1, unsafe_allow_html=True)

    st.markdown("## How it works: \n ")

    html_message2 = """
    <p style='font-family: Arial; font-size: 20px;'>Simply input 
    \n <p style='font-family: Arial; font-size: 20px;'> 1. your car's brand 
    \n <p style='font-family: Arial; font-size: 20px;'> 2. engine size (cc) 
    \n <p style='font-family: Arial; font-size: 20px;'> 3. horsepower (bhp)
    \n <p style='font-family: Arial; font-size: 20px;'> 4. mileage (kmpl) 
    \n <p style='font-family: Arial; font-size: 20px;'> Our system will provide you with an accurate estimate of its potential selling value.</p>
    """
    st.markdown(html_message2, unsafe_allow_html=True)

    st.markdown("## Ready to get started? \n ")

    html_message3 = """
    <p style='font-family: Arial; font-size: 20px;'
    >Join us today and let us help you maximize your profit. 😀 </p>
    """
    st.markdown(html_message3, unsafe_allow_html=True)
