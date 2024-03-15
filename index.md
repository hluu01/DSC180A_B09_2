# Using Predictive and Generative AI to Detect Truthfulness in News Articles
<!--To create line break: use 2 spaces after a line or use <br>-->
Lohit Geddam [lgeddam@ucsd.edu](mailto:lgeddam@ucsd.edu)

Nicholas Shor [nshor@ucsd.edu](mailto:nshor@ucsd.edu)

Irisa Jin [irjin@ucsd.edu](mailto:irjin@ucsd.edu)

Henry Luu [hluu@ucsd.edu](mailto:hluu@ucsd.edu)

Mentor: Dr. Ali Arsanjani [arsanjani@google.com](mailto:arsanjani@google.com)

**Demo Video**
<iframe width="560" height="315" src="https://www.youtube.com/embed/EYTyIaHGdk4" frameborder="0" allowfullscreen></iframe>

<!--https://youtu.be/EYTyIaHGdk4-->

**Introduction/Background**  
Throughout the internet, there are countless sources of news that people use every day to keep themselves updated on current events. The most prevalent are news websites and social media apps. These platforms have grown consistently as web applications have continued to develop, making them the primary news sources for people worldwide. The problem is that with this change, it has become easier than ever for misinformative news to spread rapidly without being fact-checked. It is extremely difficult to detect wrong from right when the credibility of a piece of news, among other factors, cannot be easily detected. The goal of this project is to consider as many of the factors present and use these to paint a picture of where a news article may be truthful or misinformative. This will allow the user to take this information and make their best judgment on whether the news article is valid or not. Removing misinformation is crucial for protecting public health, democracy, and social cohesion by preventing the spread of false beliefs that can lead to harmful decisions and damage our trust in institutions. By promoting accurate information, we can empower individuals to make informed choices and create a more resilient and cohesive society. 

In this project we developed both Generative and Predictive models that were able to produce either truthfulness scores from 1-100, or return a truthfulness label. The labels we used comes from PolitiFact’s Truth-o-meter scale which has a range of classifications which are, True, Mostly-true, Half-true, Barely-true, False, and Pants-on-fire. This gives a much more accurate description of the truth value of an article in comparison to a simple binary classification stating whether something is true or false. The truthfulness of a statement is never simply true or false and is instead a blurred scale where many factors must be accounted for. In the end, our model was able to accurately predict the label of a statement from this six way classification over 82\% of the time. Also, a majority of the incorrect predictions were only one label off which is very promising. This is still far from where we would like the final product to be to fully implement a project like this, but in a field where much work still must be done it is a very impressive result.

**Dataset**

## Data

The data utilized in this project comprised a combination of pre-existing datasets from previous studies and additional data collected through web scraping for specific project requirements. The dataset section is categorized into predictive and generative AI data, corresponding to the distinct datasets used for these methodologies.

### Predictive AI Data

#### Liar Liar Plus Dataset
The primary dataset employed was the Liar Liar Plus dataset [1], encompassing statements made by individuals or posted online, along with associated truth labels, subjects, speakers, speaker’s occupations, political affiliations, statement locations, and supplementary justifications. This dataset served as the foundational dataset for training the autoML and BERT full-text embedding models (see Figure 1).

#### Sean Jiang's Scraped PolitiFact Dataset
Additionally, we utilized a dataset scraped by a classmate [2], containing truth ratings from PolitiFact for various speakers. This dataset provided information on the truthfulness of statements categorized into six levels: true, mostly true, half true, barely true, false, and pants on fire. It facilitated the training of n-gram, autoML, sentiment analysis, quality of writing, and sensationalism models (see Figure 2).

#### Kaggle Clickbait Dataset
Furthermore, we incorporated a Clickbait dataset obtained from Kaggle [3], consisting of article headlines labeled as clickbait (1) or non-clickbait (0). This dataset was utilized to train the clickbait detection model.

### Generative AI Data

#### Politifact.com Data
We collected information from Politifact.com [4], a Pulitzer Prize-winning fact-checking website, comprising two datasets: one containing article titles and text, and another containing statements made by individuals along with corresponding truth values and explanations for the ratings (see Tables 1 and 2).

#### FactCheck.org Data
Data was also scraped from FactCheck.org [5], another reputable fact-checking organization, providing articles' text, titles, dates, and additional list data (see Table 3).

#### SciCheck.org Data
From SciCheck.org [6], a division of FactCheck.org focusing on false scientific claims, we obtained datasets containing article text, titles, and dates (see Table 4).

#### Science.Feedback.org Data
Finally, we acquired data from Science.Feedback.org [7], a fact-checking organization specializing in scientific claims, consisting of statements and corresponding labels (see Table 5).

**Methodology**

**Results**

**Discussion**


