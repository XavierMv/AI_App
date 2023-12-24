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
    tab_titles = ['Numpy', 'Pandas']
    tab1, tab2 = st.tabs(tab_titles)

    with tab1:
        st.header('Numpy')
        st.write('As a first step install and import numpy for usage')
        st.write("Run the first commmand on your terminal and add the next chunk of code on the top of your script.")

        install_code = """
        pip install numpy
        """
        st.code(install_code, language = "python")

        # Chunk of code for importing NumPy
        st.write("### Import NumPy")
        import_code = """
        import numpy as np
        """
        st.code(import_code, language = "python")

        # Chunk of code for basic operations with NumPy
        st.write("### Basic Operations with NumPy")
        st.write("See how you can use python for basic mathematical operations with numpy bult-in functions.")
        operations_code = """
        # Let's start by creating a numpy array.
        arr = np.array([1, 2, 3, 4, 5])

        # Print the array to see the information that is embedded in it.
        print(arr)

        # Calculate the mean of the array
        mean_value = np.mean(arr)
        print("Mean:", mean_value)

        # Calculate the sum of the array
        sum_value = np.sum(arr)
        print("Sum:", sum_value)
        """
        st.code(operations_code, language = "python")


    with tab2:
        st.header('Pandas')
        st.write('As a first step install and import pandas for usage')
        st.write("Run the first commmand on your terminal and add the next chunk of code on the top of your script.")

        install_code = """
        pip install pandas
        """
        st.code(install_code, language = "python")

        # Chunk of code for importing NumPy
        st.write("### Import Pandas")
        import_code = """
        import pandas as pd
        """
        st.code(import_code, language = "python")


    
def machine_learning_page():
    st.title("Machine Learning Algorithms")
    selected_approach = st.radio("Select a Machine Learning Approach", ["Unsupervised", "Supervised"])

    if selected_approach == "Unsupervised":
        st.write("Let's create a simple unsupervised machine learning model using the Iris dataset.")
        selected_algorithm = st.radio("Select an Unsupervised Machine Learning Algorithm", ["K-Means", "Hierarchical Clustering"])
        if selected_algorithm == "K-Means":
            st.write("Here is the code for K-Means")
        elif selected_algorithm == "Hierarchical Clustering":
            st.write("Here is the code for Hierarchical Clustering")

    elif selected_approach == "Supervised":
        st.write("Let's explore Supervised Machine Learning.")
        selected_supervised_model = st.radio("Select a Supervised Machine Learning Model", ["Ridge", "Lasso"])
        if selected_supervised_model == "Ridge":
            st.write("Here is the code for Ridge")
        elif selected_supervised_model == "Lasso":
            st.write("Here is the code for Lasso")
    
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Homepage": homepage,
        "Basic Python": basic_python_page,
        "Machine Learning": machine_learning_page,
    }

    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    pages[selected_page]()

if __name__ == "__main__":
    main()
