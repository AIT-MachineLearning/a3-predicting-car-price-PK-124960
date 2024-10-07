def set_page_config():
    
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
    
import streamlit as st
import os
from streamlit_option_menu import option_menu
from LinearClass import Linear, LinearPenalty
from LinearClassA3 import Ridge, RidgePenalty
import home, New_version, Newest_version
   
set_page_config()
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
                menu_title='PONKRIT st.124960 v4.0.19',
                options=['Home','New version','Newest version'],
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
        # if app == "A1 version":
        #     Old_version.main()    
        if app == "New version":
            New_version.main() 
        if app == "Newest version":
            Newest_version.main()       
             
    run()            
         