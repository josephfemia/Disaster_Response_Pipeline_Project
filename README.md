# Installations
To run the scripts you will need Python 3. The libraries that are used within the scripts include numpy, pandas, json, plotly, nltk, flask, sqlalchemy, and sklearn. Make sure that you install all libraries by running:
pip install [libraries name here]

Another way to install is by downloading and using Anaconda. This is my preferred way because it comes with the necessary libraries as well as jupyter notebook already installed rather than installing all one by one.

# Motivation
I was motivated to find out how I can help those in tough situations during emergencies. Once an emergency occurs, social media gets flooded with information. I wanted to create something that can filter through all of the messages made by people and select the messages that actually relate to those people needed some sort of help. 

I created a NLP Pipeline to try and filter out messages based on certain words used in each message. I then created a dashboard so that people can interact with the site and eve type in their own messages to see how the program will react to them.

# Descriptions
- app - Contains the information needed to load up and run the website
- data - Contains the data used to train the model as well as the cleaning techniques used to clean the data
- models - Contains the model used to filter the messages
- categories.csv - data used in the ETL Pipeline Preparation and ML Pipeline Preparation jupyter notebooks
- messages.csv - data used in the ETL Pipeline Preparation and ML Pipeline Preparation jupyter notebooks

# How to use the Project
In order to be able to see the webpage, as well as the visuals, you will need to host the webpage either locally or with a third party. To run the hosting files everything is stored in app/run.py. Manipulation will need to be made for different hosting platforms.

In order to see the cleaning and modeling process step by step check the ETL Pipeline Preparation and ML Pipeline Preparation jupyter notebooks.

# Authors
By: Joseph Femia

# Acknowledgements
This dataset was provided by Figure Eight. I wanted to say thank you to Figure Eight for providing this dataset. Also, thank you to Udacity, for promoting this dataset. 
