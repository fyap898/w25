# Assignment 3: Classification of Text Data
Due March ~~21~~ 25, 2025

You may work in teams of 2 or 3. Click [here](https://classroom.github.com/a/nIgSPAa3) to create your team on GitHub Classroom.

## Table of Contents <!-- omit in toc -->
- [Overview](#overview)
- [Dataset](#dataset)
    - [Enabling the BigQuery API](#enabling-the-bigquery-api)
- [Deliverables](#deliverables)
    - [Resources](#resources)
    - [Your code](#your-code)
    - [Your report](#your-report)
- [Marking Scheme](#marking-scheme)

## Overview
The purpose of this assignment is to further your hands-on experience in neural networks with a new type of data: text! 

## Dataset
The dataset for this assignment is the [Stack Overflow dataset](https://console.cloud.google.com/marketplace/product/stack-exchange/stack-overflow), an archive of all Stack Overflow questions, answers, and metadata since 2009, updated on a quarterly basis. This is a massive dataset that is impractical to load all at once; you will be using Google's [BigQuery API](https://cloud.google.com/bigquery/docs/reference/rest) to query the database and select a sample of the data.

### Enabling the BigQuery API
Google Cloud is rather overwhelming and provides a ton of services. For this project, we only need the BigQuery API.
1. Go to https://console.cloud.google.com/ 
2. Click on the "mtroyal.ca" link and select "New Project"
3. Make sure your new project is active, then click on the hamburger menu at the top left and navigate to "APIs and Services" -> "Enabled APIs and Services". 
4. Search for "BigQuery API" and select "Enable"

The free tier of Google Cloud provides up to 1 TB per month of data access, which *should* be plenty for this assignment. 

## Deliverables
Your assignment should consist of the following:
1. Your notebook(s) and/or Python scripts where you did your experiments, with the final training run and evaluation rendered
2. A report describing your experiments and your final model decisions
3. Your final model in [`.keras`](https://keras.io/guides/serialization_and_saving/) format, or similar if you're using a different framework. **Please make sure that it loads and runs properly using the default package versions in Colab**.

### Resources
GPU resources are a challenge, and potentially more important with text data. Here are a few options to consider:
- [Kaggle](https://www.kaggle.com/code) provides 30 hrs/week of free GPU usage
- [Colab](https://colab.research.google.com/signup) provides a "pay as you go" tier, which is $14 for 100 compute units with 90 day expiration. Colab Pro is also an option at $14/month. I hate asking students to pay for things, but think of it like the cost of a textbook. I'm testing out the pay as you go option, and it seems to be using up credits at roughly 1.5/hr on a T4 TPU.

### Your code
You are not required to use Tensorflow and Keras, but if you use different packages, please make sure that it runs on Colab, including the necessary commands to `!pip install`.

The data loading is a bit tricky for this one - I've included some sample code in the `starter.ipynb` to get you started. If you're not using Colab and/or tensorflow, you might need to massage things a bit more (yes, this ends up being an inordinate amount of time in any data project).

### Your report
In a separate document, summarize your experiments, models, observations, reflections, etc. I've provided a template with more details in the starter code (`report.md`), though you aren't limited to the markdown format.

## Marking Scheme
As before, I'll be marking on the following 4-point scale. I've given up on the "evidence of collaboration" piece, as Colab + Notebooks + git = chaos.

| Component                                               | Weight |
| ------------------------------------------------------- | ------ |
| Report: Model development and experimentation           | 25%    |
| Report: reflections                                     | 25%    |
| Report: abstract and appendices                         | 25%    |
| Model: performance and compatibility                    | 25%    |

| Score | Description                                                            |
| ----- | ---------------------------------------------------------------------- |
| 4     | Excellent - thoughtful and creative without any errors or omissions    |
| 3     | Pretty good, but with minor errors or omissions                        |
| 2     | Mostly complete, but with major errors or omissions, lacking in detail |
| 1     | A minimal effort was made, incomplete or incorrect                     |
| 0     | No effort was made, or the submission is plagiarized                   |

