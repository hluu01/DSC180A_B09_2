# Using Predictive and Generative AI to Detect Truthfulness in News Articles

**UCSD Halıcıoğlu Data Science Institute**

**UC San Diego Data Science Senior Capstone Project**
<!--To create line break: use 2 spaces after a line or use <br>-->

[Lohit Geddam](mailto:lgeddam@ucsd.edu), [Nicholas Shor](mailto:nshor@ucsd.edu), [Irisa Jin](mailto:irjin@ucsd.edu), [Henry Luu](mailto:hluu@ucsd.edu)

Mentor: [Dr. Ali Arsanjani](mailto:arsanjani@google.com)

## Table of Contents

- [Introduction/Background](#Introduction/Background)
- [Methodology](#Methodology)
- [Demo](#Demo-Video)
- [Test-It-Yourself](#Give-it-a-test-yourself!-:D)
- [Data](#Data)
- [Results](#Results)
- [Discussion](#Discussion)

## Introduction/Background
In this project we developed both Generative and Predictive models that were able to produce either truthfulness scores from 1-100, or return a truthfulness label. The labels we used comes from PolitiFact’s Truth-o-meter scale which has a range of classifications which are, True, Mostly-true, Half-true, Barely-true, False, and Pants-on-fire. This gives a much more accurate description of the truth value of an article in comparison to a simple binary classification stating whether something is true or false. The truthfulness of a statement is never simply true or false and is instead a blurred scale where many factors must be accounted for. In the end, our model was able to accurately predict the label of a statement from this six way classification over 82\% of the time. Also, a majority of the incorrect predictions were only one label off which is very promising. This is still far from where we would like the final product to be to fully implement a project like this, but in a field where much work still must be done it is a very impressive result.



## Methodology

**Here is how our final model works:**

![Here is how our final model works](/images/Dsc_capstone_flowchart_final_model.png)

The methodology employed in this project encompasses both Predictive and Generative AI approaches to tackle the issue of assessing truthfulness in news articles. Notably, significant emphasis was placed on leveraging state-of-the-art language models and AI techniques to achieve accurate evaluations.

## Demo Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/EYTyIaHGdk4" frameborder="0" allowfullscreen></iframe>

<!--https://youtu.be/EYTyIaHGdk4-->

# [Give-it-a-test-yourself!-:D](http://774c35acd4cc675e18.disinformation-destroyers.com/?fbclid=IwAR3fxMeCWVS2XrxoO32GWxLe95UK3BThBCgPxctN17W8BX2WRopofzyb2gw)

### Predictive AI Models we developed:

- **Full Text N-gram Analysis with Logistic Regression**

- **Full Text BERT Analysis with Random Forest Classification**

- **Readability Metric using Flesch-Kincaid Readability Score**

- **Clickbait Detection using Support Vector Classification**

- **Quality of Writing Detection using GaussianNB**

- **Sensationalism Detection using GaussianNB**

- **Sentiment Analysis using NLTK Sentiment Package and GaussianNB**

- **Trained Models using Google Vertex AI AutoML**

### Generative AI

The Generative AI process commenced with the instantiation of Google's Gemini Pro model through the Google Cloud Vertex AI console. Configurations were applied to customize the model for the specific task, including setting safety configurations to ensure the output's integrity. RAG, a powerful NLP technique, was employed to enhance the Generative AI process. We utilized a vector database populated with embeddings of fact-checked statements and news articles from reputable sources. An Approximate Nearest Neighbors Search (ANN) algorithm facilitated the retrieval of relevant information, enhancing the model's contextual understanding. Extensive prompt engineering was undertaken to refine the model's input instructions. Specific and detailed prompts were crafted to guide the model in providing accurate and logical responses. Chain-of-Thought reasoning was incorporated to ensure systematic thinking and coherent output.

### Final Deployment
The final deployment involved integrating various AI approaches to provide comprehensive evaluations of news articles. Chunking of articles, combined with context enrichment through Generative AI models, contributed to accurate assessments. Gradio, a user-friendly web interface platform, was utilized for model deployment, enabling users to receive truthfulness evaluations along with predictive scores and overall article synopses.

## Data

The data utilized in this project comprised a combination of pre-existing datasets from previous studies and additional data collected through web scraping for specific project requirements. The dataset section is categorized into predictive and generative AI data, corresponding to the distinct datasets used for these methodologies.

### Predictive AI Data

- **Politifact**
![Politifact Data](/images/politifact_data.jpg)

- **SciCheck.org**
![SciCheck.org](/images/scicheck_data.png)

- **Liar Liar Plus - Politics**

- **FactCheck.org - US Politics**

- **ScienceFeedback.org - Healthcare**

## Results

### Predictive Model Results
### Generative Model Results

## Discussion

### Interpretation of the Results & Other Works

The results obtained from our predictive and generative AI models showcase promising advancements in truthfulness detection within news articles. Notably, our predictive models demonstrated varying levels of accuracy across different techniques, with Google Vertex AI AutoML showcasing particularly impressive performance. Furthermore, our generative AI approach, especially when augmented with retrieval techniques and prompt engineering, exhibited substantial improvements in accuracy compared to baseline methods.

These findings align with recent advancements in AI and NLP, indicating the potential of machine learning models in addressing the pervasive issue of misinformation and disinformation online. Moreover, our work contributes to a growing body of literature aimed at enhancing media and digital literacy, thereby fostering a more informed and discerning readership.

### Impact & Limitations

Because accurate information is critical for making informed decisions, this project contributes to combating misinformation and disinformation online, providing a faster and more efficient method for verifying online sources. By offering an improvement in media and digital literacy, particularly for readers from diverse educational backgrounds, our model has the potential to enhance technological accessibility and foster trust in journalism.

Despite these contributions, there are several limitations to consider. Firstly, the scale and scope of our project constrained the size and domain of the data used. While we employed various datasets, primarily focusing on American politics and science-related articles, the availability of manually fact-checked data restricted the diversity of our sources. Future research could benefit from expanding the dataset size and diversity to encompass a broader range of topics and regions.

Moreover, while our model achieved an accuracy of 82.4%, it is essential to recognize that truthfulness assessment is inherently subjective and context-dependent. Additional improvements in processing power and prompt engineering may further enhance model performance. Ultimately, our model serves as a valuable tool for truthfulness assessment, but definitive conclusions about the absolute truthfulness of an article or statement remain elusive, given the multiplicity of perspectives on any given topic. Continued refinement and validation of our methodologies are necessary to address these challenges and foster a more nuanced understanding of truthfulness in media content.



