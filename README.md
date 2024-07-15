# Knowledge Representation and Insights Generation from Structured Dataset

This project focuses on developing an AI-based solution for knowledge representation and insight generation from structured datasets. It encompasses several key functionalities:

1. Data Pre-processing: Cleaning and preparing datasets by handling missing values, normalizing data, and encoding categorical variables.

2. Knowledge Representation: Using visual tools to create intuitive and informative visualizations, such as histograms, box plots, scatter plots, and heatmaps.

3. Pattern Identification: Applying machine learning algorithms like PCA, KMeans, and DBSCAN to detect patterns, trends, and anomalies within the dataset.

4. Insight Generation: Utilizing natural language generation models to convert analysis results into human-readable insights and recommendations.

5. User-friendly Interface: Providing an interactive web-based interface using Flask, allowing users to upload datasets, view visualizations, and receive generated insights.

---

## Installation (To Run the Project):

1. Clone the repository

2. Open **Complete Code** folder on your editor and follow the instructions given below:
    
3. Set Up a Virtual Environment
   It is recommended to use a virtual environment to manage dependencies but it is not necessary when you install dependencies file **requirements.txt**.
   You can set up a virtual environment using venv:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
5. Install Required Packages
   Install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```
   Ensure requirements.txt includes all dependencies such as Flask, Pandas, Matplotlib, Seaborn, Scikit-learn, and Transformers.

6. Install Additional Dependencies
   Make sure you have all the required libraries for running the project.

7. For start the application please run the following command

   ```bash
   python app.py
   ```
8. Open your web browser and navigate to http://127.0.0.1:5000/ to start using the application.

---

## Note:
   1. Do not click on Data Preprocessing Option again because data is preprocess at once.
   2. Make sure that you have proper internet connection to run the project otherwise it may be timeout from browser or may be delay for processing the output.
   
## Project Structure
   The project structure is as follows:

    Complete Code 
    ├── app.py                 
    ├── templates/              
    │   ├── index.html
    │   ├── upload.html
    │   ├── output.html
    │   ├── preprocessing.html
    │   ├── knowledge_representation.html
    │   ├── pattern_identification.html
    │   ├── insight_generation.html
    │   ├── about_us.html
    │   ├── contact_us.html
    ├── static/
    │   ├── images/
    │   │   ├── background.jpeg
    │   │   ├── pranav.jpeg
    │   │   ├── Yash.jpg
    │   │   ├── Yash_d.jpg
    │   │   ├── rr_karwa.jpeg
    │   │   ├── au_chaudhari.jpg
    │   ├── styles/
    │   │   ├── style.css
    │   │   ├── style_out.css
    ├── uploads
    │   ├──Life Expectancy Data 
    └── requirements.txt  

---

## **About the Team**:
### Team Name : *Codecrafters*
### Team Lead:
  1. [Pranav Rajput](https://github.com/24-Pranav)
### Team Members:
  2. [Yash Lawankar](https://github.com/devloperYash)
  3. [Yash Dighade](https://github.com/Hitman45-web)

## **Reach out our youtube channel for project tutorial**

https://youtu.be/k0nNWaFWbWA

Special thanks to **Intel Unnati Industrial Training Program 2024**
