from flask import Flask, request, render_template,redirect, url_for, flash
from markupsafe import Markup
import pandas as pd
import io
import numpy as np
import matplotlib
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from markupsafe import Markup
from multiprocessing import Pool
from transformers import pipeline

app = Flask(__name__)

app.secret_key = 'CodeCrafters'

#  Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#  To hold the uploaded data
df = None  
preprocessed_df = None 

def preprocess_data(df):
    numerical_cols = df.select_dtypes(include="number").columns

    # Impute missing values before encoding
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(df[numerical_cols])
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols)
    df[numerical_cols] = imputed_df
    df = df.dropna()

    # Outlier detection and treatment using whisker method
    def whisker(col):
        q1, q2 = np.percentile(col, [25, 75])
        iqr = q2 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q2 + (1.5 * iqr)
        return lower_bound, upper_bound

    for col in numerical_cols:
        lower, upper = whisker(df[col])
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    # Encode categorical variables
    categorical_features = ['Country']
    df = pd.get_dummies(df, columns=categorical_features)

    # Label encoding for 'Status' column
    label_encoder = LabelEncoder()
    if 'Status' in df.columns:
        df['Status'] = label_encoder.fit_transform(df['Status'])

    return df


# Cache directory for storing plot images
CACHE_DIR = './cache'
plt.style.use('ggplot')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

selected_columns = ['Year', 'Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                    'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 
                    'Polio', 'Total expenditure', 'Diphtheria ', ' HIV or AIDS', 'GDP', 'Population',
                    ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling']

def save_plot(fig, filename):
    filepath = os.path.join(CACHE_DIR, filename)
    fig.savefig(filepath, format='png')
    with open(filepath, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{img_data}'

def plot_histogram(df_col):
    col, df = df_col
    if col not in selected_columns:
        return None
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=col, ax=ax, color='darkorange')
    ax.set_title(f'Distribution of {col}')
    return save_plot(fig, f'hist_{col}.png')

def plot_boxplot(df_col):
    col, df = df_col
    if col not in selected_columns:
        return None
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=col, ax=ax, color='mediumseagreen', width=0.5) 
    ax.set_title(f'Boxplot of {col}')
    return save_plot(fig, f'box_{col}.png')

def plot_scatter(df_cols):
    x_col, y_col, df = df_cols
    if x_col == y_col or x_col not in selected_columns or y_col not in selected_columns:
        return None
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color='tomato')
    ax.set_title(f'Relationship between {x_col} and {y_col}')
    return save_plot(fig, f'scatter_{x_col}_{y_col}.png')

def plot_heatmap(corr_matrix):
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    return save_plot(fig, 'heatmap.png')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global df
    if request.method == 'POST':
        file = request.files['file']
        if file:
            #  Save the file to the 'uploads' directory
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            #  Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
           
            return redirect(url_for('overview'))
    return render_template('upload.html')


@app.route('/overview')
def overview():
    global df
    if df is None:
        df=pd.read_csv("uploads\Life Expectancy Data.csv")
    
    data_head = df.head(10).to_html()
    data_tail = df.tail(10).to_html()
    shape = df.shape

    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    Data_type = df.dtypes.to_frame('dtypes').to_html()
    
    return render_template('output.html', data_head=Markup(data_head), data_tail=Markup(data_tail),shape=shape, info=info.replace('/n','<b>'), Data_type=Data_type )


@app.route('/preprocess')
def preprocess():
    global df, preprocessed_df
    if df is None:
        df=pd.read_csv("uploads\Life Expectancy Data.csv")
    
    missing_values = df.isnull().sum().to_frame('count').to_html()
    missing_percentage = (df.isnull().sum() / df.shape[0] * 100).to_frame('percentage').to_html()
   
    duplicate_rows = df.duplicated().sum()
    garbage_values = df.select_dtypes(include=['object']).apply(lambda x: x[x.str.contains('garbage', case=False)].count()).to_list()
   
    numerical_stats = df.select_dtypes(include=['int64', 'float64']).describe().to_html()
    categorical_stats = df.select_dtypes(include=['object']).describe().to_html()

    numerical_cols = df.select_dtypes(include="number").columns

    #  Impute missing values before encoding
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(df[numerical_cols])
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols)
    df[numerical_cols] = imputed_df
    df = df.dropna()

    no_missing_values = df.isnull().sum().to_frame('count').to_html()
    
    #  Outlier detection and treatment using whisker method
    def whisker(col):
        q1, q2 = np.percentile(col, [25, 75])
        iqr = q2 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q2 + (1.5 * iqr)
        return lower_bound, upper_bound

    outliers_count = {}
    for col in numerical_cols:
        lower, upper = whisker(df[col])
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outliers_count[col] = len(outliers)
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    
    #  Encode categorical variables
    categorical_features = ['Country']
    df = pd.get_dummies(df, columns=categorical_features)

    #  Label encoding for 'Status' column
    label_encoder = LabelEncoder()
    if 'Status' in df.columns:
        df['Status'] = label_encoder.fit_transform(df['Status'])

    encoded_data = df.head(10).to_html()

    #  Calculate the null count
    null_count = df.isnull().sum()

    #  Shape of the dataset after encoding
    shape_after_encoding = df.shape    
    
    #  Store the preprocessed dataset globally
    preprocessed_df = df.copy()

    print(f"Preprocessed DataFrame shape: {preprocessed_df.shape}")  # Debugging info

    return render_template('preprocessing.html', 
                           missing_values=missing_values, 
                           missing_percentage=missing_percentage, 
                           duplicate_rows=duplicate_rows,
                           garbage_values=garbage_values, 
                           numerical_stats=numerical_stats, 
                           categorical_stats=categorical_stats,
                           null_count=null_count, 
                           encoded_data=encoded_data, 
                           no_missing_values=no_missing_values, 
                           shape_after_encoding=shape_after_encoding,
                           outliers_count=outliers_count)


@app.route('/knowledge_representation')
def knowledge_representation():
    global df
    global selected_columns
    
    if preprocessed_df is None:
        df=pd.read_csv("uploads\Life Expectancy Data.csv")
        # Preprocess the data
        df = preprocess_data(df)
    
    # Validate selected_columns
    if not selected_columns:
        selected_columns = df.columns.tolist()  # Fallback to all columns if none are selected

    cols_to_plot = [col for col in selected_columns if col in df.columns]
    
    if not cols_to_plot:
        return redirect(url_for('upload'))

    histogram_data = [(col, df) for col in cols_to_plot]
    boxplot_data = [(col, df) for col in cols_to_plot]
    scatter_data = [(x_col, 'Life expectancy ', df) for x_col in cols_to_plot if x_col != 'Life expectancy ']

    with Pool() as pool:
        hist_images = pool.map(plot_histogram, histogram_data)
        box_images = pool.map(plot_boxplot, boxplot_data)
        scatter_images = pool.map(plot_scatter, scatter_data)

    heatmap_img_1 = plot_heatmap(df[cols_to_plot].corr())

    return render_template('knowledge_representation.html', hist_images=hist_images, box_images=box_images, 
                           scatter_images=scatter_images, heatmap_img_1=heatmap_img_1)


@app.route('/pattern_identification')
def pattern_identification():
    global df
    if preprocessed_df is None:
        df=pd.read_csv("uploads\Life Expectancy Data.csv")
        # Preprocess the data
        df = preprocess_data(df)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    df['pca_one'] = pca_result[:, 0]
    df['pca_two'] = pca_result[:, 1]

    elbow_plot = ''
    wcss = []
    for i in range(1, 11):
         kmeans = KMeans(n_clusters=i, random_state=42)
         kmeans.fit(df)
         wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    plt.plot(range(1, 11), wcss, marker='o')
    ax.set_title('Elbow Method for Optimal K')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    elbow_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df[['pca_one', 'pca_two']])

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='pca_one', y='pca_two', hue='cluster', palette='viridis', ax=ax)
    ax.set_title('KMeans Clusters')
    kmeans_clusters = save_plot(fig, 'kmeans_clusters.png')

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(df[['pca_one', 'pca_two']])

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='pca_one', y='pca_two', hue='dbscan_cluster', palette='viridis', ax=ax)
    ax.set_title('DBSCAN Clusters')
    dbscan_clusters = save_plot(fig, 'dbscan_clusters.png')

    # Linear Regression
    X = df.drop(columns=['Life expectancy '])
    y = df['Life expectancy ']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    y_train_pred = lin_reg.predict(X_train)
    y_test_pred = lin_reg.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)

    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_r2 = r2_score(y_test, y_test_pred)

    linear_regression_results = f"Train RMSE: {train_rmse}, Train R2: {train_r2}, Test RMSE: {test_rmse}, Test R2: {test_r2}"

    # Random Forest Regression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_train)

    y_train_pred_rf = rf_reg.predict(X_train)
    y_test_pred_rf = rf_reg.predict(X_test)

    train_rmse_rf = mean_squared_error(y_train, y_train_pred_rf, squared=False)
    train_r2_rf = r2_score(y_train, y_train_pred_rf)

    test_rmse_rf = mean_squared_error(y_test, y_test_pred_rf, squared=False)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)

    random_forest_results = f"Train RMSE: {train_rmse_rf}, Train R2: {train_r2_rf}, Test RMSE: {test_rmse_rf}, Test R2: {test_r2_rf}"


    return render_template('pattern_identification.html', elbow_plot=elbow_plot,linear_regression_results=linear_regression_results, 
                                   kmeans_clusters=kmeans_clusters, dbscan_clusters=dbscan_clusters, random_forest_results=random_forest_results)


@app.route('/insight_generation')
def insight_generation():
    global preprocessed_df

    # Handle the case where preprocessed_df is None
    if preprocessed_df is None:
        try:
            df = pd.read_csv('uploads/Life Expectancy Data.csv')
            preprocessed_df = preprocess_data(df)  # Ensure this function is defined elsewhere
        except FileNotFoundError:
            return redirect(url_for('upload'))
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            return render_template('insight_generation.html', error=str(e))
    
    try:
        # Split the data into features and target
        X = preprocessed_df.drop('Life expectancy ', axis=1)  # Adjust the column name to match your dataset
        y = preprocessed_df['Life expectancy ']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a RandomForest model
        model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced the number of trees to make it faster
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mae_test = mean_absolute_error(y_test, y_pred)
        mse_test = mean_squared_error(y_test, y_pred)

        # Calculate feature importance
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

        # Load a pre-trained language model for text generation
        text_gen_model = pipeline('text-generation', model='distilgpt2')

        # Insight Generation Prompt
        insight_prompt = f"""
            Based on the analysis of the life expectancy data, the key findings are as follows:

            - The top features positively correlated with life expectancy are {', '.join(importance_df.head(5)['Feature'])}.
            - The top features negatively correlated with life expectancy are {', '.join(importance_df.tail(5)['Feature'])}.
            - The model achieved the following performance metrics on the test set:
                - Mean Absolute Error (MAE): {mae_test:.2f}
                - Mean Squared Error (MSE): {mse_test:.2f}

            Detailed insights and recommendations based on these findings:

            Top Positive Correlations:
            - {importance_df.head(1)['Feature'].values[0]}
            - {importance_df.head(2).tail(1)['Feature'].values[0]}
            - {importance_df.head(3).tail(1)['Feature'].values[0]}
            - {importance_df.head(4).tail(1)['Feature'].values[0]}
            - {importance_df.head(5).tail(1)['Feature'].values[0]}

            Top Negative Correlations:
            - {importance_df.tail(1)['Feature'].values[0]}
            - {importance_df.tail(2).head(1)['Feature'].values[0]}
            - {importance_df.tail(3).head(1)['Feature'].values[0]}
            - {importance_df.tail(4).head(1)['Feature'].values[0]}
            - {importance_df.tail(5).head(1)['Feature'].values[0]}

            Recommendations:
            - Analyze and address the most significant positive and negative correlations with life expectancy to improve overall health outcomes.
            - Focus on features like {importance_df.head(1)['Feature'].values[0]} and {importance_df.tail(1)['Feature'].values[0]} to target interventions.

            Ensure to avoid repetition and provide concise and actionable insights.
        """

        # Generate insights
        insights = text_gen_model(insight_prompt, max_new_tokens=200)[0]['generated_text']

        # Post-process the generated insights
        insights = insights.replace('\n', '<br>').replace('<br><br>', '<br>')

        # Correlation Analysis
        corr_matrix = preprocessed_df.corr()
        top_positive_corr = corr_matrix['Life expectancy '].sort_values(ascending=False).head(5).to_frame('Top Positive Correlations')
        top_negative_corr = corr_matrix['Life expectancy '].sort_values().head(5).to_frame('Top Negative Correlations')

        # Distribution of key features
        key_image = []
        key_features = importance_df.head(5)['Feature'].tolist()
        for i, feature in enumerate(key_features):
            plt.figure(figsize=(6, 4))  #  Reduced the figure size to make it faster
            sns.histplot(preprocessed_df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            key_image.append(f'data:image/png;base64,{img_data}')
            plt.close()


        return render_template('insight_generation.html', insights=Markup(insights),
                               top_positive_corr=top_positive_corr.to_html(classes='table table-striped', border=0),
                               top_negative_corr=top_negative_corr.to_html(classes='table table-striped', border=0),
                               key_image=key_image)
    except Exception as e:
        print(f"Error during insight generation: {e}")
        return render_template('insight_generation.html', error=str(e))




@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')


if __name__ == '__main__':
    app.run(debug=True)