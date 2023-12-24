import streamlit as st
import pandas as pd
import numpy as np 


def homepage():
    st.title("Welcome to My Streamlit App")
    st.write("In this streamlit App I want to give you a cheatsheet of how to use Python for AI applications in an easy way.")
    # Add more content about what users can find on your app

def basic_python_page():
    st.title("Basic Python Commands with Pandas and NumPy")
    # Add content about basic Python commands, Pandas, and NumPy

def machine_learning_page():
    st.title("Machine Learning Algorithms")
    # Add content about machine learning algorithms and source code

def main():
    st.sidebar.title("Navigation")
    pages = {
        "Homepage": homepage,
        "Basic Python": basic_python_page,
        "Machine Learning": machine_learning_page,
    }

    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    # Display the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
