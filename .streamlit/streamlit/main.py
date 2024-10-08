import streamlit as st
from streamlit_option_menu import option_menu
from LinearClass import *
import os
# from dotenv import load_dotenv
# load_dotenv()

import home, Old_version, New_version

st.set_page_config(
    page_title="ML @AIT.SET",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='PONKRIT st.124960',
                options=['Home','Old version','New version'],
                icons=['house-fill','person-circle','person-circle'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        if app == "Home":
            home.app()
        if app == "Old version":
            Old_version.main()    
        if app == "New version":
            New_version.main()       
             
    run()            
         