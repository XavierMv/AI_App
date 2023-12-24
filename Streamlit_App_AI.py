import streamlit as st
import pandas as pd
import numpy as np 


def homepage():
    st.title("Welcome to My Streamlit App")
    st.write(
        "Hello and welcome to my Streamlit app! My goal is to provide you with "
        "basic knowledge on how to use Python for AI and business applications."
    )

    st.markdown(
        """
        ## Professional Background
        I bring a strong background in credit risk analysis and data science, with hands-on experience
        in developing impactful solutions for leading financial and consulting organizations.

        ### Santander - Credit Risk Intern
        As a Credit Risk intern at Santander, I played a key role in developing time series models to estimate
        credit default. This involved leveraging advanced statistical techniques to analyze historical data and
        predict future credit events. My work contributed to enhancing the bank's ability to make informed
        decisions in managing credit risk.

        ### Accenture - Data Scientist
        In my role as a Data Scientist at Accenture, I worked on a variety of projects spanning revenue growth
        management (RGM), data monetization strategies, customer analytics, and smart spend strategies. I provided
        valuable consulting advice for AI implementations across different industries. My responsibilities included
        developing models to optimize RGM, crafting data monetization strategies, and providing insights through
        advanced analytics.

        My experience at Accenture involved collaborating with cross-functional teams, interpreting complex data
        sets, and delivering actionable insights to clients. I also contributed to the development of innovative AI
        solutions tailored to the specific needs of diverse industries.

        My educational background includes studies in Actuarial Sciences and financial engineering at Universidad
        Anáhuac México. Currently, I am pursuing a Master of Management in Artificial Intelligence at Queen's University.
        This academic journey has provided me with a solid foundation in both traditional and cutting-edge techniques
        in the fields of finance and artificial intelligence.

        I am passionate about the intersection of data science and business strategy, and I am excited to share my
        knowledge with you through this Streamlit app.
        """
    )

    st.markdown(
        """
        ## Connect with Me
        If you have any questions or would like to connect, feel free to reach out to me on LinkedIn:
        """
    )

    st.markdown("[Connect with Me on LinkedIn](https://www.linkedin.com/in/xaviermagana/)")


def basic_python_page():
    st.title("Basic Python Commands with Pandas and NumPy")
    # Add content about basic Python commands, Pandas, and NumPy

def machine_learning_page():
    st.title("Machine Learning Algorithms")
    # Add content about machine learning algorithms and source code

def main():
    # Add navigation bar at the top
    st.beta_container()
    st.markdown(
        """
        <style>
            .reportview-container {
                flex-direction: row;
            }

            header > .toolbar {
                flex-direction: row;
                width: 100%;
                padding-left: 10px;
                padding-right: 10px;
            }

            header > .toolbar > * {
                flex: 1;
            }
        </style>
        """
    )

    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction_page,
        "Basic Python": basic_python_page,
        "Machine Learning": machine_learning_page,
    }

    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    # Display the selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
