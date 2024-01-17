import streamlit as st
import pandas as pd
import numpy as np 
import openai


def homepage():
    st.title("Welcome to My Streamlit App")
    st.write(
        "My goal is to provide you with "
        "basic knowledge on how to use Python for AI and business applications."
    )

    st.markdown(
        """
        ## Professional Background
        I bring a strong background in credit risk analysis and data science, with hands-on experience
        in developing impactful solutions for leading financial and consulting organizations.

        ### Accenture - Data Scientist
        In my role as a Data Scientist at Accenture, I worked on a variety of projects spanning revenue growth
        management (RGM), data monetization strategies, customer analytics, and smart spend strategies. I provided
        valuable consulting advice for AI implementations across different industries. My responsibilities included
        developing models to optimize RGM, crafting data monetization strategies, and providing insights through
        advanced analytics.

        My experience at Accenture involved collaborating with cross-functional teams, interpreting complex data
        sets, and delivering actionable insights to clients. I also contributed to the development of innovative AI
        solutions tailored to the specific needs of diverse industries.

        ### Santander - Credit Risk Intern
        As a Credit Risk intern at Santander, I played a key role in developing time series models to estimate
        credit default. This involved leveraging advanced statistical techniques to analyze historical data and
        predict future credit events. My work contributed to enhancing the bank's ability to make informed
        decisions in managing credit risk.

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


def basic_programming():
    st.title("Basic programming with R and Python")
    tab_titles = ['Overview', 'Coding']
    tab1, tab2 = st.tabs(tab_titles)

    with tab1:
        st.header('General Overview')

        st.markdown("""
        Welcome to a foundational section on data manipulation! Mastering basic programming commands in R and Python 
        is a crucial step before diving into AI and advanced data manipulation. The provided code snippets will be 
        essential in upcoming sections, empowering you to explore statistical analyses and AI applications. 
        Understanding these fundamentals is a strategic investment, unlocking the potential of your data science 
        journey. Let's delve into the synergy between programming and the dynamic fields of statistics and AI, propelling 
        your skills to new heights!
        """)
        


    with tab2:
        st.header('Start with the basics')

        selected_topic = st.selectbox("Select an option:", ["Declare variables", "Vectors and Matrices", "Dataframes", "Loops","Plots"])

        if selected_topic == "Declare variables":
            selected_language = st.radio("Select a programming language:", ["R", "Python"])
            if selected_language == "R":
                st.markdown("""
                ## Declare variables
                Let's start with the basics, how to assign values to variables
                """)
                assign_r = """
                # Assign values to variables 
                x <- 5
                y <- 9

                # See the type of variable that you have
                class(x)
                class(y)
                """

                st.code(assign_r, language = "python")

                st.markdown("""
                ## Math operations
                Try the following lines of code to see how to do basic math with R
                """)

                math_r = """
                # Do some basic math
                x + y 
                x - y 
                x*y 
                x/y
                x^y
                sin(x)
                cos(x)
                tan(x)
                """

                st.code(math_r, language = "python")
                
                
            elif selected_language == "Python":

                st.markdown("""
                ## Declare variables
                Let's start with the basics, how to assign values to variables
                """)

                assign_py = """
                # Assign values to variables 
                x = 5
                y = 9

                # See the type of variable that you have
                print(type(x), type(y))
                """

                st.code(assign_py, language = "python")


                st.write("Let's start using some libraries for math operations")
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

                math_py = """
                # Do some basic math
                print(x + y)
                print(x - y)
                print(x*y)
                print(x/y)
                print(x^y)
                print(np.sin(x))
                print(np.cos(x))
                print(np.tan(x))
                """

                st.code(math_py, language = "python")

                
            
            
        elif selected_topic == "Vectors and Matrices":
            
            selected_language = st.radio("Select a programming language:", ["R", "Python"])
            if selected_language == "R":
                st.markdown("""
                ## Vectors and matrices
                Try the following lines of code to see how to do vectors and matrices with R
                """)

                v_m_r = """
                # Declare vectors through different commands
                z <- 1:10
                w <- c(4,2,5,7,8,9,0,3,2,6)
                x <- runif(10)*100
                t <- sample(x,5, replace = TRUE)
                y <- seq(from=10, to=1, by=-1)
                z
                w
                x
                t
                y

                # Let's create a matrix of 4 columns and 10 rows with vectors (x,z,w,y)
                mat <- matrix(data= c(x,z,w,y), ncol = 4, nrow = 10)
                mat

                """

                st.code(v_m_r, language = "python")

                st.markdown("""
                ## Operations with vectors and matrices
                Try the following lines of code to see the math between these objects
                """)

                vmo_r = """
                # Declare the following objects
                mata <- matrix(1:9, ncol = 3, nrow = 3)
                matb <- matrix(10:18, ncol = 3, nrow = 3)
                va <- 1:3
                vb <- 4:6
                mata
                matb
                va
                vb


                # Do some math
                mata + matb
                mata - matb
                mata / matb
                mata * matb
                sin(mata)
                mata^2
                mata + va
                mata + vb
                mata*va
                mata/va
                va*vb
                va + vb
                va - vb

                """

                st.code(vmo_r, language = "python")

            elif selected_language == "Python":
                st.markdown("""
                ## Vectors and matrices
                Try the following lines of code to see how to do vectors and matrices with python
                """)

                v_m_py = """
                # Declare vectors through different commands
                z = np.arange(1, 11)
                w = np.array([4, 2, 5, 7, 8, 9, 0, 3, 2, 6])
                x = np.random.rand(10) * 100
                t = random.choices(x, k=5)
                y = np.arange(10, 0, -1)
                print(z)
                print(w)
                print(x)
                print(t)
                print(y)

                # Let's create a matrix of 4 columns and 10 rows with vectors (x,z,w,y)
                mat = np.column_stack((x, z, w, y))
                print(mat)
                """

                st.code(v_m_py, language = "python")

                st.markdown("""
                ## Operations with vectors and matrices
                Try the following lines of code to see the math between these objects
                """)

                vmo_py = """
                # Declare the following objects
                mata = np.reshape(np.arange(1, 10), (3, 3))
                matb = np.reshape(np.arange(10, 19), (3, 3))
                va = np.arange(1, 4)
                vb = np.arange(4, 7)
                print(mata)
                print(matb)
                print(va)
                print(vb)


                # Do some math
                print(mata + matb)
                print(mata - matb)
                print(mata / matb)
                print(mata * matb)
                print(np.sin(mata))
                print(mata**2)
                print(mata + va)
                print(mata + vb)
                print(mata*va)
                print(mata/va)
                print(va*vb)
                print(va + vb)
                print(va - vb)

                """
                st.code(vmo_py, language = "python")

        elif selected_topic == "Dataframes":

            selected_language = st.radio("Select a programming language:", ["R", "Python"])

            if selected_language == "R":
                st.markdown("""
                ## Tidyverse for data manipulation
                            
                Let's start by installing the library on the terminal
                """)
                install_df_r = """
                install.packages("tidyverse")
                """
                st.code(install_df_r, language = "python")

                st.write("Read Tidyverse library. This command should be at the top of your script")
                tidy_r = """
                library(tidyverse)
                """
                st.code(tidy_r, language = "python")

                st.write("Work with dataframes")
                dfm_r = """
                # Declare vectors through different commands
                z <- 1:10
                w <- c(4,2,5,7,8,9,0,3,2,6)
                x <- runif(10)*100
                y <- seq(from=10, to=1, by=-1)

                # Create a matrix from the previous vectors
                mat <- matrix(data= c(x,z,w,y), ncol = 4, nrow = 10)
                colnames(mat) <- c("x","z","w","y")
                mat

                # Turn it into a dataframe
                df <- data.frame(mat) 
                df
                """
                st.code(dfm_r, language = "python")



            elif selected_language == "Python":
                st.markdown("""
                ## Pandas for data manipulation
                            
                Let's start by installing the library on the terminal
                """)
                install_pandas_py = """
                pip install pandas
                """
                st.code(install_pandas_py, language = "python")

                st.write("Import Pandas")
                import_pandas_py = """
                import pandas as pd
                """
                st.code(import_pandas_py, language = "python")

                st.write("Work with dataframes")
                dfm_r = """
                # Declare vectors through different commands
                z = np.arange(1, 11)
                w = np.array([4, 2, 5, 7, 8, 9, 0, 3, 2, 6])
                x = np.random.rand(10) * 100
                y = np.arange(10, 0, -1)

                # Create a matrix from the previous vectors
                mat = np.column_stack((x, z, w, y))

                # Turn it into a dataframe
                col_names = ["x", "z", "w", "y"]
                df = pd.DataFrame(mat, columns=col_names)
                print(df)
                """
                st.code(dfm_r, language = "python")
        

        

        

def statistical_learning_page():
    st.title("Statistical Learning Algorithms")
    tab_titles_sl = ['General Overview', 'Regression', 'Classification']
    GO, regression, classification = st.tabs(tab_titles_sl)

    with GO:
        st.markdown(
            """
            Statistical learning is a mathematical approach for supervised learning models. In supervised learning models, 
            we have two types of features:

                - Independent features (denoted by X values): These features aim to explain and provide insight into the
                dependent feature.
                - Dependent feature (denoted by Y): This feature is the one we are trying to predict/estimate based on the
                other features.

            This statistical approach involves conducting a series of parametric tests to ensure the performance and
            reliability of the model for prediction. The model can be fitted with different probability distributions
            depending on the type of answer we are seeking. For the purpose of this app, we will use the regression model
            that assumes normality and logistic regression, which assumes fitness from the binomial distribution.

            In statistical learning models, it is necessary to split the dataset into training and testing sets due to the
            continuous validation of parameters throughout the model-building steps. 

            It is important that you follow the following steps for model evaluation:

                1. Plot each dependent feature to the independent feature to understand the function
                2. Look at summary table (F-statistic, features' p-values, normality test, Autocorrelation test, r-squared, and adjusted r-squared)
                3. Look for qq-plot and normality test
                4. Look for homocedasticity (constant variance among residuals)
                5. Look for FIV values (the threshold depends on the business requirement)

            The regression equation is denoted as follows:

            """)
        
        regression_equation = r"Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon"
        st.latex(regression_equation)

        
    
    with regression:
        st.write("## Regression Analysis")
        st.write("Let's start by importing certain libraries and functions that will help us in further steps")        
        libraries_rm = """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_regression
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        import scipy as sp
        import statsmodels.api as sm
        """
        st.code(libraries_rm, language = "python")

        st.write("Split the model into training and testing and add the constant value")
        split_rm = """
        X = sm.add_constant(X)
        X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=.3, train_size=None, random_state=1, 
                                                          shuffle=True, stratify=None)
        """
        st.code(split_rm, language = "python")

        st.write("Train the model and get a summary table for model evaluation")
        st.markdown("""
        The following evaluation will be separated into steps:
            
            - Look for the p-value of the F-statistic to be less than 0.05 to ensure your model is valid
            - Review the p-values of each feature introduced in the model. Every feature most be under 0.05 or be removed due to lack of contribution to the model prediction
            - Look for the Durbin-Watson test for non-autocorrelation. We are looking for a value around 2
            - Look for the r-squared to understand the proportion of variance that's being explained.
            
        """)
        split_rm = """
        ols_m = sm.OLS(y_train,X_train).fit()
        print(ols_m.summary())
        y_pred = ols_m.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        y_avg = y.mean()
        print("y average:", y_avg)
        print("MSE:",mse)
        print("SQRT MSE:", np.sqrt(mse))
        print("MAE:",mae)
        """
        st.code(split_rm, language = "python")

        st.write("""
        Create a qq-plot to make sure the residuals do have a normal distribution, you can apply normality tests 
        to your analysis (Anderson-Darling, Shapiro-Wilks, among others).
        """)

        qqplot_rm = """
        error = y_test-y_pred
        fig, ax = plt.subplots(figsize=(4,4),dpi=100)
        _=sp.stats.probplot(error,plot=ax)
        """

        st.code(qqplot_rm, language = "python")

        normality_rm = """

        # Shapiro-Wilks test
        from scipy.stats import shapiro

        stat, p_value = shapiro(error)

        if p_value > 0.05:
            print("There's normality")
        else:
            print("There's no normality")

        
        # Anderson-Darling test
        from scipy.stats import anderson

        result = anderson(error)

        print("Test Statistic: ", result.statistic)
        
        """

        st.code(normality_rm, language = "python")

        st.write("""
        Once you have evaluated each test, make sure to make adjustments to the model with representative features 
        and retrain it for performance improvement.
        """)
        
        

    
    with classification:
        st.write("## Logistic Regression")

    
def machine_learning_page():
    st.title("Machine Learning Algorithms")
    tab_titles = ['General Overview', 'Unsupervised', 'Supervised']
    GOML, Unsupervised, Supervised = st.tabs(tab_titles)

    with GOML:
        st.markdown("""
        
        Machine Learning is a part of AI, it takes the same logic from statistical learning models to create 
        models that can be faster, run with less tests and enhance it's predictions. These type of models are less interpretable 
        and more flexible, you can find 2 basic groups:
        
        - Unsupervised: These models do not have an objective feature Y, you're trying to find patterns in your data to create groups and explain them
        - Supervised: These models do have an objective feature that can be either continous (Regression) or categorical (Classification)
        
        This type of models need to be divided on Train, Test, and Cross-Validation sets to ensure a good fit for the model.
        All this models have a set of hyperparameters that need to change to enhance it's performance, this process is called hypertuning and can be 
        hard to deliver on local computers, depending on the size of the dataset and the type of model you're trying to train.
        
        """)

    with Unsupervised:
        
        selected_algorithm = st.selectbox("Select an Unsupervised Machine Learning Algorithm", ["K-Means", "Hierarchical Clustering"])
        if selected_algorithm == "K-Means":
            st.markdown("""
            K-Means algorithm works through the following steps:
                        
            1. Scale the features
            2. Create distance matrix (Confusion matrix is embedded on KMeans function from scikit-learn)
            3. Evaluate optimal number of clusters through Elbow method & Silhouette method
            4. Choose the optimal number of clusters and assess through business criteria
            
            """)

            selected_language = st.radio("Select a programming language:", ["R", "Python"])
            if selected_language == "R":
                
                kmeans_r = """
                # Upload the necessary libraries
                library(ggplot2)
                library(dendextend)
                library(factoextra)
                library(cluster)

                # Scale the data and create a distance matrix
                df <- scale(data)
                dist_mat <- dist(df, method = "euclidean")

                # Create Elbow method and Silhouette
                fviz_nbclust(dist_mat, FUN = kmeans, method = "wss")
                fviz_nbclust(dist_mat, FUN = kmeans, method = "silhouette")

                # Select the optimal number of clusters, let's assume is 4
                clust <- kmeans(dist_mat, 4)
                clust$cluster
                """

                st.code(kmeans_r, language = "python")

            elif selected_language == "Python":

                kmeans_py = """
                # Import libraries
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                import matplotlib.pyplot as plt

                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Generate Elbow method
                inertia_values = []
                for k in range(1, 11):
                    kmeans = KMeans(n_clusters = k, random_state = 42)
                    kmeans.fit(X_scaled)
                    inertia_values.append(kmeans.inertia_)

                # Plot Elbow method
                plt.plot(range(1, 11), inertia_values, marker = 'o')
                plt.title('Elbow Method for Optimal k')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Inertia')
                plt.show()

                # Generate Silhouette method
                silhouette_scores = []
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters = k, random_state = 42)
                    labels = kmeans.fit_predict(X_scaled)
                    silhouette_scores.append(silhouette_score(X_scaled, labels))

                # Plot Silhouette method
                plt.plot(range(2, 11), silhouette_scores, marker = 'o')
                plt.title('Silhouette Method for Optimal k')
                plt.xlabel('Number of Clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.show()

                # Select the optimal number of clusters for the final model
                final_k = 4
                kmeans_final = KMeans(n_clusters = final_k, random_state = 42)
                labels = kmeans_final.fit_predict(X_scaled)

                # Visualize the clusters
                plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c = labels, cmap = 'viridis', edgecolor = 'k')
                plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], c = 'red', marker = 'X', s = 200)
                plt.title("KMeans Clustering with k = " + str(final_k))
                plt.xlabel('Feature 1 (Scaled)')
                plt.ylabel('Feature 2 (Scaled)')
                plt.show()

                """

                st.code(kmeans_py, language = "python")
            

            

        elif selected_algorithm == "Hierarchical Clustering":
            st.markdown("""
            Hierarchical Clustering algorithm works through the following steps:
                        
            1. Scale the features
            2. Computing linkage matrix
                - Ward: Minimize the variance within the instances from the cluster
                - Complete: Distance between clusters is the maximum distance between individual instances
                - Single: Distance between clusters is the minimum distance between individual instances
                - Average: Distance between clusters is the average distance between individual instances
            3. Evaluate optimal number of clusters through Elbow method, Silhouette method & Dendogram plot
            4. Choose the optimal number of clusters and assess through business criteria
            
            """)
            selected_language = st.radio("Select a programming language:", ["R", "Python"])
            if selected_language == "R":
                
                hierarchical_r = """
                # Upload the necessary libraries
                library(ggplot2)
                library(dendextend)
                library(factoextra)
                library(cluster)

                # Scale the data
                df <- scale(data)
                Hmodel <- hclust(df, method = "ward.D")

                # Create Elbow method and Silhouette
                fviz_nbclust(dist_mat, FUN = hcut, method = "wss")
                fviz_nbclust(dist_mat, FUN = hcut, method = "silhouette")

                # Plot Dendogram
                plot(Hmodel)

                # Select the optimal number of clusters, let's assume is 4
                clust <- cutree(Hmodel, 4)
                clust$cluster
                """

                st.code(hierarchical_r, language = "python")

            elif selected_language == "Python":

                hierarchical_py = """

                # Import libraries
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import AgglomerativeClustering
                from sklearn.metrics import silhouette_score
                import matplotlib.pyplot as plt
                import plotly.figure_factory as ff

                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Generate Elbow method
                max_clusters = 12 
                distortions = []
                for k in range(1, max_clusters + 1):
                    model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
                    model.fit(X_scaled)
                    distortions.append(np.sum((X_scaled - X_scaled.mean(axis = 0))**2) / X_scaled.shape[0])


                # Plot Elbow method
                plt.plot(range(1, max_clusters + 1), distortions, marker = 'o')
                plt.title('Elbow Method')
                plt.xlabel('Number of clusters')
                plt.ylabel('Total within Sum of Squares')
                plt.show()

                # Generate Silhouette method
                silhouette_scores = []
                for k in range(2, max_clusters + 1):
                    model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
                    clusters = model.fit_predict(X_scaled)
                    silhouette_scores.append(silhouette_score(X_scaled, clusters))


                # Plot Silhouette method
                plt.plot(range(2, max_clusters + 1), silhouette_scores, marker = 'o')
                plt.title('Silhouette Method')
                plt.xlabel('Number of clusters')
                plt.ylabel('Score')
                plt.show()

                # Plot Dendogram
                fig = ff.create_dendrogram(X_scaled)
                fig.update_layout(width = 800, height = 500)
                fig.show()


                # Select the optimal number of clusters for the final model and let's assume is 4
                hierarchical_4 = AgglomerativeClustering(n_clusters = 4, linkage='ward')
                clusters_4 = hierarchical_4.fit_predict(X_scaled)
                """

                st.code(hierarchical_py, language = "python")

            


    with Supervised:
        st.write("Let's explore Supervised Machine Learning.")
        selected_supervised_model = st.selectbox("Select a Supervised Machine Learning Model", ["Ridge", "Lasso", "LDA", "SVM", "Decision Tree", "Random Forest", "GBM", "XGBM", "LightGBM"])
        if selected_supervised_model == "Ridge":
            st.markdown("""
            Ridge regression, a machine learning approach akin to linear regression in statistical learning, involves fitting your data while retaining all available features.
            To identify the optimal model, pinpoint the critical juncture where errors escalate exponentially. In doing so, seek a model incorporating a specific penalization 
            factor around that pivotal area. This meticulous process ensures the creation of a robust and effective model.
            """)
            
            rdige_code = """

            # Import libraries
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import GridSearchCV, train_test_split
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_squared_error
            from sklearn.preprocessing import StandardScaler

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

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

            # Import libraries
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import GridSearchCV, train_test_split
            from sklearn.linear_model import Lasso
            from sklearn.metrics import mean_squared_error
            from sklearn.preprocessing import StandardScaler

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Define the Lasso Regression model
            lasso_model = Lasso()

            # Define the hyperparameter grid. This you can select if you want to run the GridSearch through all the hyperparameters
            param_grid = {
                'alpha': np.logspace(-3, 3, 7),  
                'normalize': [True, False],        
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator = lasso_model, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5)

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
            lda_code = """
            # Import libraries
            import pandas as pd
            import numpy as np
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.model_selection import GridSearchCV, train_test_split
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

            # Define the Linear Discriminant Analysis model
            lda_model = LinearDiscriminantAnalysis()

            # Define the hyperparameter grid for GridSearchCV
            param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto'],
                'n_components': [None, 1, 2],  # Number of components for dimensionality reduction
            }

            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator = lda_model, param_grid = param_grid, scoring = 'accuracy', cv = 5)

            # Fit the model
            grid_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = grid_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Evaluate the best model on test data
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on Test Set:", accuracy)

            # Performance Metrics
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))
            """

            st.code(lda_code, language = "python")

        elif selected_supervised_model == "SVM":
            st.write("Here is the code for SVM")
            svm_code = """

            # Import libraries
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
            grid_search = GridSearchCV(estimator = svm_model, param_grid = param_grid, scoring = 'accuracy', cv = 5)

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
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))


            """
            st.code(svm_code, language = "python")


        elif selected_supervised_model == "Decision Tree":
            st.write("Here is the code for Decision Tree")
            DT_code = """
            import pandas as pd
            import numpy as np
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import RandomizedSearchCV, train_test_split
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

            # Define the Decision Tree model
            dt_model = DecisionTreeClassifier()

            # Define the hyperparameter grid for Random Search
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': np.arange(1, 21),
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': np.arange(1, 11),
                'max_features': [None, 'auto', 'sqrt', 'log2'],
            }

            # Create Random Search object
            random_search = RandomizedSearchCV(estimator = dt_model, param_distributions = param_grid, scoring = 'accuracy', cv = 5, n_iter = 50, random_state = 42)

            # Fit the model
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = random_search.best_estimator_

            # Evaluate the best model on test data
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on Test Set:", accuracy)

            # Performance Metrics
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))

            """
            st.code(DT_code, language = "python")

        elif selected_supervised_model == "Random Forest":
            st.write("Here is the code for Random Forest")
            rf_code = """
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import RandomizedSearchCV, train_test_split
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            # Assuming you have your dataset X and y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

            # Define the Random Forest model
            rf_model = RandomForestClassifier()

            # Define the hyperparameter grid for Random Search
            param_grid = {
                'n_estimators': np.arange(50, 251, 20),
                'criterion': ['gini', 'entropy'],
                'max_depth': np.arange(1, 21),
                'min_samples_split': np.arange(2, 11),
                'min_samples_leaf': np.arange(1, 11),
                'max_features': ['auto', 'sqrt', 'log2'],
                'bootstrap': [True, False],
            }

            # Create Random Search object
            random_search = RandomizedSearchCV(estimator = rf_model, param_distributions = param_grid, scoring = 'accuracy', cv = 5, n_iter = 50, random_state = 42)

            # Fit the model
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = random_search.best_estimator_

            # Evaluate the best model on test data
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on Test Set:", accuracy)

            # Performance Metrics
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))

            """

            st.code(rf_code, language = "python")

        elif selected_supervised_model == "GBM":
            st.write("Here is the code for GBM")
            gbm_code = """
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import RandomizedSearchCV, train_test_split
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

            # Assuming you have your dataset X and y for classification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

            # Define the GBM model
            gbm_model = GradientBoostingClassifier()

            # Define the hyperparameter grid for RandomizedSearchCV
            param_grid = {
                'n_estimators': np.arange(50, 200, 10),
                'learning_rate': np.logspace(-3, 0, 4),
                'max_depth': np.arange(3, 10),
                'min_samples_split': np.arange(2, 10),
                'min_samples_leaf': np.arange(1, 10),
            }

            # Create RandomizedSearchCV object
            random_search = RandomizedSearchCV(estimator = gbm_model, param_distributions = param_grid, n_iter = 50, scoring = 'accuracy', cv = 5, random_state = 42)

            # Fit the model
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Get the best model
            best_model = random_search.best_estimator_

            # Evaluate the best model on test data
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print("Accuracy on Test Set:", accuracy)
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))

            """
            st.code(gbm_code, language = "python")


        elif selected_supervised_model == "XGBM":
            st.write("Here is the code for XGBM")
            xgbm_code = """

            # Import libraries
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

            # Import libraries
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

def nlp():
    st.title("Natural Language Processing")
    tab_titles = ['Overview', 'ChatGPT API', 'Sentiment Analysis', 'Topic Models', 'Transformers']
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    with tab1:
        st.header('Overview')
    
    with tab2:
        st.markdown("""
        The first thing you need to do to get started with ChatGPT on Python is to log-in on the following link:
                    
        https://platform.openai.com/api-keys
        
        Go to the *API Keys* tab and create a new key and save it, we will use it in further steps to access ChatGPT.
        
        Type the following code on your python script and get a full interaction with ChatGPT.
                    
        *NOTE: The calls to the API will be limited on your billing plan*
        """)
    
        chat_gpt_code = """

        import openai
        import os

        os.environ["OPENAI_API_KEY"] = "Write your API key"

        client = openai.OpenAI()

        text_user = input("User: ")

        response = client.chat.completions.create(
        model="gpt-3.5-turbo", # You can change the version of GPT depending on your billing plan
        messages=[
            {"role": "user", "content": text_user}
            ]
        )
        
        print(f'ChatGPT: {response.choices[0].message.content}')
        """

        st.code(chat_gpt_code, language = "python")

    
def main():
    st.sidebar.title("Navigation")
    pages = {
        "About Me": homepage,
        "Basic Programming for Statistics & AI": basic_programming,
        "Statistical Learning": statistical_learning_page,
        "Machine Learning": machine_learning_page,
        "NLP": nlp
    }

    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    pages[selected_page]()

if __name__ == "__main__":
    main()
