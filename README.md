# DSC180A_B09_2
This is the code and data we used to create our predictive and generative models for our Winter 2024 UCSD Data Science Capstone project. All necessary files are located in the main branch.

Reproducing the project:

Step 1: Acquiring Data
-The first thing you will need to do is to clone this repository to your local device and then download/unzip the necessary files. There are 8 total datasets that we used and they are all located in the work_progress folder. They are called factcheckorg_webscrape.csv, nick_clickbait_data.csv, politifact_articles.csv, scicheck_data.csv, science_feedback.csv, train.tsv, politifact_truthometer_df.csv.zip, and politifact_data_combined_prev_ratings.csv.zip. Once those are unzipped, I would put them all back in a folder called Data, so that it is easy to change the file path to load the datasets.

Step 2: Creating 180B-B09-2 environment

-The next step is to create the conda environment with all the necessary dependencies to run the ipynb file I've provided. To do this you will need to open up your terminal on your local device. Make sure you have also downloaded the environment.yml file to your local device as well. You will navigate to where your environment.yml is in the terminal and then run the following line of code.

-conda env create -f environment.yml

-This will create a conda environment for you where you will have all the necessary dependencies. From here, you can open a jupyter notebook or whatever environment you usually program in and open the FullModelPipeline.ipynb file provided.

Step 3: Running FullModelPipeline.ipynb

-Running the FullModelPipeline.ipynb should be fairly straightforward but there are a couple of things to keep in mind. Firstly, if for any reason a package wasn't properly downloaded you can do a simple Google search to find the pip installation and it should work with the latest versions of all packages as of March 14th, 2024. Once this is done we can get to the rest of the code.

-All you need to do is run the code line by line to reproduce the results we obtained. The code is split up into many individual sections. All sections of the code should run fairly quickly except for a couple of sections. The training of the predictive models in the Predictive Models section will likely take you about 45 minutes. Additionally, the uploading of data to the Chroma Vector Database step in the Vector Database section takes about 3-4 hours as there is a huge amount of contextual data that needs to be uploaded to the database for the Generative AI model to refer to. Additionally, it is important to note that you will need a Google Cloud account and Perspective API keys to fully reproduce this project. We also have connected our Gradio web implementation to a website domain that we purchased from GoDaddy so this is a necessary step as well to have a permanent website domain. All other steps should run appropriately assuming that you have properly cloned all the contents of this repository. This concludes everything you need to do to reproduce this project. Thank you.
