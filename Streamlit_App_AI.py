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
        mean_val = np.mean(arr)
        print("Mean:", mean_val)

        # Calculate the sum of the array
        sum_val = np.sum(arr)
        print("Sum:", sum_val)
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
    tab_titles = ['Unsupervised', 'Supervised']
    tab1, tab2 = st.tabs(tab_titles)

    with tab1:
        st.write("Let's create a simple unsupervised machine learning model using the Iris dataset.")
        selected_algorithm = st.selectbox("Select an Unsupervised Machine Learning Algorithm", ["K-Means", "Hierarchical Clustering"])
        if selected_algorithm == "K-Means":
            st.write("Here is the code for K-Means")
        elif selected_algorithm == "Hierarchical Clustering":
            st.write("Here is the code for Hierarchical Clustering")

    with tab2:
        st.write("Let's explore Supervised Machine Learning.")
        selected_supervised_model = st.selectbox("Select a Supervised Machine Learning Model", ["Ridge", "Lasso", "LDA", "SVM", "Decision Tree", "GBM", "XGBM", "LightGBM"])
        if selected_supervised_model == "Ridge":
            st.write("Here is the code for Ridge")
            rdige_code = """
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import GridSearchCV, train_test_split
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_squared_error
            from sklearn.preprocessing import StandardScaler

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Define the Ridge Regression model
            ridge_model = Ridge()

            # Define the hyperparameter grid. This you can select if you want to run the GridSearch through all the hyperparameters
            param_grid = {
                'alpha': np.logspace(-3, 3, 7),  
                'normalize': [True, False],        
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator = ridge_model, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5)

            # Fit the model
            grid_search.fit(X_train_scaled, y_train)

            # Get the best hyperparameters
            best_params = grid_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Evaluate the best model on test data
            X_test_scaled = scaler.transform(X_test)
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error on Test Set:", mse)

            """
            st.code(rdige_code, language = "python")

        elif selected_supervised_model == "Lasso":
            st.write("Here is the code for Lasso")
            lasso_code = """
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import GridSearchCV, train_test_split
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_squared_error
            from sklearn.preprocessing import StandardScaler

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Define the Ridge Regression model
            ridge_model = Lasso()

            # Define the hyperparameter grid. This you can select if you want to run the GridSearch through all the hyperparameters
            param_grid = {
                'alpha': np.logspace(-3, 3, 7),  
                'normalize': [True, False],        
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator = ridge_model, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5)

            # Fit the model
            grid_search.fit(X_train_scaled, y_train)

            # Get the best hyperparameters
            best_params = grid_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Evaluate the best model on test data
            X_test_scaled = scaler.transform(X_test)
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error on Test Set:", mse)

            """
            st.code(lasso_code, language = "python")


        elif selected_supervised_model == "LDA":
            st.write("Here is the code for LDA")


        elif selected_supervised_model == "SVM":
            st.write("Here is the code for SVM")
            svm_code = """
            import sklearn
            from sklearn.svm import SVC
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import classification_report
            from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
            from sklearn.model_selection import train_test_split, GridSearchCV
            import pandas as pd
            import numpy as np

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

            svm_model = SVC()

            # Define the hyperparameter grid for GridSearchCV
            param_grid = {
                'C': np.logspace(-3, 3, 7),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4],
            }

            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring='accuracy', cv=5)

            # Fit the model
            grid_search.fit(X_train_scaled, y_train)

            # Get the best hyperparameters
            best_params = grid_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Evaluate the best model on test data
            X_test_scaled = scaler.transform(X_test)
            y_pred = best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on Test Set:", accuracy)

            # Performance Metrics
            print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
            print("Classification Report:\n", classification_report(y_test, pred))

            """
            st.code(svm_code, language = "python")


        elif selected_supervised_model == "Decision Tree":
            st.write("Here is the code for Decision Tree")
        elif selected_supervised_model == "Random Forest":
            st.write("Here is the code for Random Forest")
        elif selected_supervised_model == "GBM":
            st.write("Here is the code for GBM")


        elif selected_supervised_model == "XGBM":
            st.write("Here is the code for XGBM")
            xgbm_code = """
            import pandas as pd
            import numpy as np
            import xgboost as xgb
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import classification_report
            from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
            from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

            clf = xgb.XGBClassifier()

            # Change the parameters inside the brackets to hypertune

            # Look for model documentation in the following link -> https://xgboost.readthedocs.io/en/stable/parameter.html 
                
            param_grid = {
                'max_depth': np.arange(10,50,10),
                'learning_rate': np.arange(0.1,0.9,0.1),
                'n_estimators': np.arange(50,250,20), 
                'gamma': np.arange(0.1,0.5,0.1),
                'min_child_weight': np.arange(0.3,1,0.1),
                'subsample': [0.5], 
                'colsample_bytree': np.arange(0.2,0.8,0.2),
                'reg_alpha': [0.5], 
                'reg_lambda': [0.5],
                'device': ['gpu']}

            # Create RandomSearch object                
            random_search = RandomizedSearchCV(clf, param_distributions = param_grid, n_iter = 50, scoring = 'accuracy', n_jobs = 1, cv = 5, random_state = 42)

            # Fit the model
            random_search.fit(X_train, y_train)

            # Get the best model
            best_model = random_search.best_estimator_
            print(best_model)

            # Evaluate the best model on test data
            X_test_scaled = scaler.transform(X_test)
            y_pred = best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on Test Set:", accuracy)

            # Performance Metrics
            print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
            print("Classification Report:\n", classification_report(y_test, pred))

            """
            st.code(xgbm_code, language = "python")

        elif selected_supervised_model == "LightGBM":
            st.write("Here is the code for LightGBM")
            lightgbm_code = """
            import pandas as pd
            import numpy as np
            import lightgbm as lgb
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import classification_report
            from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
            from sklearn.model_selection import train_test_split, GridSearchCV

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

            
            clf = lgb.LGBMClassifier()

            # Change the parameters inside the brackets to hypertune
            # Look for model documentation in the following link -> https://lightgbm.readthedocs.io/en/latest/Parameters.html 
            param_grid = {
                'boosting_type' :  ['gbdt', 'dart', 'goss'],
                'num_leaves' : [3000, 5000],
                'max_depth' : [0],
                'learning_rate' : [0.1, 0.3, 0.5],
                'n_estimators' : [50, 100],
                'min_child_weight' : [0.001], 
                'subsample' : [1],
                'colsample_bytree' : [0.5], 
                'reg_alpha' : [0.5], 
                'reg_lambda' : [0.5], 
                'device' : ['gpu']}

            # Create GridSearch object
            grid_search = GridSearchCV(clf, param_grid, cv = 5, scoring = 'accuracy', verbose = 1)

            # Fit the model
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_
            print(best_model)

            # Evaluate the best model on test data
            X_test_scaled = scaler.transform(X_test)
            y_pred = best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on Test Set:", accuracy)

            # Performance Metrics
            print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
            print("Classification Report:\n", classification_report(y_test, pred))

            """
            st.code(lightgbm_code, language = "python")


    
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
