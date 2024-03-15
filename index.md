# Using Predictive and Generative AI to Detect Truthfulness in News Articles

**Collaborators**
<!--To create line break: use 2 spaces after a line or use <br>-->
Lohit Geddam [lgeddam@ucsd.edu](mailto:lgeddam@ucsd.edu)

Nicholas Shor [nshor@ucsd.edu](mailto:nshor@ucsd.edu)

Irisa Jin [irjin@ucsd.edu](mailto:irjin@ucsd.edu)

Henry Luu [hluu@ucsd.edu](mailto:hluu@ucsd.edu)

Mentor: Dr. Ali Arsanjani [arsanjani@google.com](mailto:arsanjani@google.com)

## Demo Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/EYTyIaHGdk4" frameborder="0" allowfullscreen></iframe>

<!--https://youtu.be/EYTyIaHGdk4-->

## Introduction/Background
Throughout the internet, there are countless sources of news that people use every day to keep themselves updated on current events. The most prevalent are news websites and social media apps. These platforms have grown consistently as web applications have continued to develop, making them the primary news sources for people worldwide. The problem is that with this change, it has become easier than ever for misinformative news to spread rapidly without being fact-checked. It is extremely difficult to detect wrong from right when the credibility of a piece of news, among other factors, cannot be easily detected. The goal of this project is to consider as many of the factors present and use these to paint a picture of where a news article may be truthful or misinformative. This will allow the user to take this information and make their best judgment on whether the news article is valid or not. Removing misinformation is crucial for protecting public health, democracy, and social cohesion by preventing the spread of false beliefs that can lead to harmful decisions and damage our trust in institutions. By promoting accurate information, we can empower individuals to make informed choices and create a more resilient and cohesive society. 

In this project we developed both Generative and Predictive models that were able to produce either truthfulness scores from 1-100, or return a truthfulness label. The labels we used comes from PolitiFact’s Truth-o-meter scale which has a range of classifications which are, True, Mostly-true, Half-true, Barely-true, False, and Pants-on-fire. This gives a much more accurate description of the truth value of an article in comparison to a simple binary classification stating whether something is true or false. The truthfulness of a statement is never simply true or false and is instead a blurred scale where many factors must be accounted for. In the end, our model was able to accurately predict the label of a statement from this six way classification over 82\% of the time. Also, a majority of the incorrect predictions were only one label off which is very promising. This is still far from where we would like the final product to be to fully implement a project like this, but in a field where much work still must be done it is a very impressive result.

## Data

The data utilized in this project comprised a combination of pre-existing datasets from previous studies and additional data collected through web scraping for specific project requirements. The dataset section is categorized into predictive and generative AI data, corresponding to the distinct datasets used for these methodologies.

### Predictive AI Data

#### Liar Liar Plus Dataset
The primary dataset employed was the Liar Liar Plus dataset, encompassing statements made by individuals or posted online, along with associated truth labels, subjects, speakers, speaker’s occupations, political affiliations, statement locations, and supplementary justifications. This dataset served as the foundational dataset for training the autoML and BERT full-text embedding models.

#### Sean Jiang's Scraped PolitiFact Dataset
Additionally, we utilized a dataset scraped by a classmate, containing truth ratings from PolitiFact for various speakers. This dataset provided information on the truthfulness of statements categorized into six levels: true, mostly true, half true, barely true, false, and pants on fire. It facilitated the training of n-gram, autoML, sentiment analysis, quality of writing, and sensationalism models.

#### Kaggle Clickbait Dataset
Furthermore, we incorporated a Clickbait dataset obtained from Kaggle, consisting of article headlines labeled as clickbait (1) or non-clickbait (0). This dataset was utilized to train the clickbait detection model.

### Generative AI Data

#### Politifact.com Data
We collected information from Politifact.com, a Pulitzer Prize-winning fact-checking website, comprising two datasets: one containing article titles and text, and another containing statements made by individuals along with corresponding truth values and explanations for the ratings.

#### FactCheck.org Data
Data was also scraped from FactCheck.org, providing article text, titles, dates, and additional data.

#### SciCheck.org Data
From SciCheck.org, a division of FactCheck.org focusing on false scientific claims, we obtained datasets containing article text, titles, and dates.

#### Science.Feedback.org Data
Finally, we acquired data from Science.Feedback.org, a fact-checking organization specializing in scientific claims, consisting of statements and corresponding labels.

## Methodology

The methodology employed in this project encompasses both Predictive and Generative AI approaches to tackle the issue of assessing truthfulness in news articles. Notably, significant emphasis was placed on leveraging state-of-the-art language models and AI techniques to achieve accurate evaluations.

### Predictive AI Models

#### Full Text N-gram Analysis with Logistic Regression
We initiated our predictive modeling process by conducting full-text n-gram analysis using Logistic Regression. This approach provided a baseline model for further exploration.

#### Full Text BERT Analysis with Random Forest Classification
Utilizing BERT embeddings extracted from the Liar Plus dataset, we employed Random Forest Classification to analyze the embeddings. The 'bert base-uncased' model from the transformers library facilitated this analysis. The GPU provided by the UCSD Data Science Machine Learning Platform (DSMLP) was instrumental in processing the embeddings efficiently.

#### Readability Metric using Flesch-Kincaid Readability Score
To evaluate the readability of news articles, we utilized the Flesch-Kincaid Grade Level score, a common metric that assesses text comprehensibility in English. This metric aids in gauging the credibility of the author and ensuring the text is suitable for the intended audience.

#### Clickbait Detection using Support Vector Classification
A clickbait detection model was trained using a dataset obtained from Kaggle, comprising labeled clickbait and non-clickbait headlines. Support Vector Classification (SVC) emerged as the best-performing algorithm for detecting clickbait content.

#### Quality of Writing Detection using GaussianNB
Gaussian Naive Bayes (GaussianNB) was employed to assess the quality of writing in news articles. Features such as Type Token Ratio (TTR) and the number of adjectives per word were used to evaluate writing quality.

#### Sensationalism Detection using GaussianNB
Similarly, Gaussian Naive Bayes (GaussianNB) was utilized to detect sensationalism in news articles. The ratio of adjectives to total words served as a key feature in this detection process.

#### Sentiment Analysis using NLTK Sentiment Package and GaussianNB
Sentiment analysis was conducted using the NLTK Sentiment Package in combination with Gaussian Naive Bayes (GaussianNB). This analysis provided insights into the overall sentiment expressed in news articles.

#### Trained Models using Google Vertex AI AutoML
To optimize predictive modeling, we leveraged Google's AutoML feature for multi-class tabular classification. This approach facilitated model selection and hyper-parameter tuning, enhancing predictive accuracy.

### Generative AI

#### Generative AI Model Instantiation
The Generative AI process commenced with the instantiation of Google's Gemini Pro model through the Google Cloud Vertex AI console. Configurations were applied to customize the model for the specific task, including setting safety configurations to ensure the output's integrity.

#### Retrieval Augmented Generation (RAG)
RAG, a powerful NLP technique, was employed to enhance the Generative AI process. We utilized a vector database populated with embeddings of fact-checked statements and news articles from reputable sources. An Approximate Nearest Neighbors Search (ANN) algorithm facilitated the retrieval of relevant information, enhancing the model's contextual understanding.

#### Prompt Engineering
Extensive prompt engineering was undertaken to refine the model's input instructions. Specific and detailed prompts were crafted to guide the model in providing accurate and logical responses. Chain-of-Thought reasoning was incorporated to ensure systematic thinking and coherent output.

### Final Deployment
The final deployment involved integrating various AI approaches to provide comprehensive evaluations of news articles. Chunking of articles, combined with context enrichment through Generative AI models, contributed to accurate assessments. Gradio, a user-friendly web interface platform, was utilized for model deployment, enabling users to receive truthfulness evaluations along with predictive scores and overall article synopses.

## Results

### Predictive Model Results

\input{table/scores}

#### Clickbait Model
Our clickbait model was different from our other predictors in the sense that it made binary predictions of 0 if the title wasn't clickbait and 1 if the title was clickbait. This is why we saw an accuracy score of 95.67%, which was much higher than the other predictive models performances.

#### BERT Embedding Model Results
Our random forest classifier trained on full-text BERT embeddings achieved a classification accuracy of 33% on the test dataset, which comprised of 1349 instances. The results indicate that while the classifier performs exceptionally well in identifying 'pants-fire' class instances with a precision of 1.00, it struggles with recall, suggesting that it often misses instances of this class.

#### Google Vertex AI AutoML
Google Vertex AI's AutoML was used to train two separate multi-class tabular classification models. The first model was trained on the Liar Liar Plus dataset and was able to achieve an overall accuracy of 50.1% and precision of 81.5% on the test data. The features that were given the most important were the credibility scores, which took into account the truthfulness of that author's past articles. The model did well in identifying articles that were either false or had little truth to it but struggled in identifying articles that were pants-fire false and true. The second model was trained on Jiang's dataset that included the entire article as a feature. This AutoML model was trained solely using the article and achieved an impressive accuracy of 95.4% based off of the area under the precision-recall curve and a precision of 91.6% on the test data. This model excelled across all labels, but showed weakness in identifying the "true" label correctly as it scored an accuracy of 88.7% for that label.

### Generative Model Results

Our experiment investigated the effectiveness of Generative and Predictive AI methods in detecting truthfulness and deceptiveness in news articles. In this section, we will be discussing how we tested our Generative AI model. Since we are working with a Generative LLM here, the process of testing the model is a bit different than traditional Machine Learning methods. What we did was tested the model by varying our prompt, and the information/context that we provided as input. This allowed us to see an improvement in performance as we made changes. To test performance we evaluated our various approaches on a randomly selected set of 500 entries from the Liar Liar dataset that were not included in our RAG. We did this by asking the model to process one of the statements from the Liar Liar dataset and then return one of the six possible labels (pants-fire, false, barely-true, half-true, mostly-true, true). We then compared the predicted labels to the actual labels we had in the dataset to test accuracy.

The first method we tested was simply feeding the statements directly into the Gemini Pro large language model. This served as our baseline, achieving an accuracy of 21%. The next step was to implement well-known Generative AI techniques to improve this performance. By simply adding Retrieval-Augmented Generation (RAG) alone, we saw a significant increase in accuracy to 59.8%. This provided the model with additional contextual information on the statement and also similar fact-checked statements on the topic. Further improvement was achieved through additional Prompt Engineering as mentioned in the Methods section. This provided the model with more guidance towards truth detection and resulted in an accuracy of 71%. Our final approach using just Generative methods was the most successful, combining RAG, our re-ranking mechanism, and prompt engineering, reaching an accuracy of 78.2%.

To further improve our model's overall performance, we also explored combining our Generative AI model with scores from our separate Predictive Models. During implementation, we saw some interesting results. When our Predictive and Generative scores were weighted equally, the accuracy dropped to 34.6%. However, by weighting the Generative AI model's score more heavily (80/20) we saw the highest overall accuracy of 82.4%. These results suggest that Generative AI methods, particularly when combined with prompt engineering and re-ranking, are promising tools for detecting truthfulness in news articles. Furthermore, incorporating scores from external models can offer additional improvements, but optimal performance relies on careful weighting of these scores and thorough testing.

## Discussion

### Interpretation of the Results & Other Works

The results obtained from our predictive and generative AI models showcase promising advancements in truthfulness detection within news articles. Notably, our predictive models demonstrated varying levels of accuracy across different techniques, with Google Vertex AI AutoML showcasing particularly impressive performance. Furthermore, our generative AI approach, especially when augmented with retrieval techniques and prompt engineering, exhibited substantial improvements in accuracy compared to baseline methods.

These findings align with recent advancements in AI and NLP, indicating the potential of machine learning models in addressing the pervasive issue of misinformation and disinformation online. Moreover, our work contributes to a growing body of literature aimed at enhancing media and digital literacy, thereby fostering a more informed and discerning readership.

### Impact & Limitations

Because accurate information is critical for making informed decisions, this project contributes to combating misinformation and disinformation online, providing a faster and more efficient method for verifying online sources. By offering an improvement in media and digital literacy, particularly for readers from diverse educational backgrounds, our model has the potential to enhance technological accessibility and foster trust in journalism.

Despite these contributions, there are several limitations to consider. Firstly, the scale and scope of our project constrained the size and domain of the data used. While we employed various datasets, primarily focusing on American politics and science-related articles, the availability of manually fact-checked data restricted the diversity of our sources. Future research could benefit from expanding the dataset size and diversity to encompass a broader range of topics and regions.

Moreover, while our model achieved an accuracy of 82.4%, it is essential to recognize that truthfulness assessment is inherently subjective and context-dependent. Additional improvements in processing power and prompt engineering may further enhance model performance. Ultimately, our model serves as a valuable tool for truthfulness assessment, but definitive conclusions about the absolute truthfulness of an article or statement remain elusive, given the multiplicity of perspectives on any given topic. Continued refinement and validation of our methodologies are necessary to address these challenges and foster a more nuanced understanding of truthfulness in media content.


