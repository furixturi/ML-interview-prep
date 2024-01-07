# Machine Learning Interview Cheatsheet

## Machine Learning

### Basics

#### 1. Explain AI, ML, and Deep Learning

AI, ML and DL are interconnectied fields, each representing different aspects and depths of the study and application of intelligent systems.

1. AI: artificial intelligence, is the broadest concept among the three. It includes any machine or software capable of performing tasks that typically require human intelligence, e.g., reasoning, learing, problem-solving, language understanding. AI can be rule-based, or it can be based on learning from data
2. ML: machine learning, is a subset of AI, focused on let machine learn decision making based on data. With machine learning, ML systems can use algorithm and data to learn to make decision or prediction, without needing to be explicitly programmed.
3. DL: deep learning is a subset of machine learning. It structures algorithm in layers to create an "artificial neural network", with which machines can learn complex patterns from large amount of data and make intelligent decisions on its own. DL is particularly good at processing large amounts of unstructured data like images, sound, text. Therefore it is used in advanced areas like CV, NLP, audio recognition.

DL is part of ML, ML is part of AI. Key difference between ML and DL is in their approach and complexity. ML can be effective with smaller datasets and work well with structured data. DL requires vast amount of training data and computation power but is effective at solving complex problems base on unstructured data, such as image and speech recognition.

#### 2. What are the different types of learning / training in ML?

There are several types of ML learning and training methods, suited to different kinds of problems and data characteristics. Some commonly used ones include:

1. **Supervised learning**: training the model on a labeled dataset, where each input is paired with the desired output. The model learns to map inputs to outputs. Use this when you have labeled data and the task is predicting outcomes, such as regression or classification problems. Example algorithms of supervised leaning are Decision Trees, Linear Regression, Logistic Regression, Support Vector Machines.
2. **Unsupervised learning**: the algorithm analyzes unlabeled data to find patterns or inherent structures without predefined labels or outcomes. This is suitable if the goal is to explore data, find hidden patterns, or reduce data dimensionality. Example algorithms: K-means clustering, Principal Component Analysis (PCA), Autoencoders.
3. **Semi-supervised Learning**: an approach that combines supervised and unsupervised learning. It uses a small mount of labeled data along with a large amount of unlabeled data during training. Use this when acquiring labeled data is expensive, but there's access to a large amount of unlabeled data. Example algorithms: semi-supervised SVM, graph-based methods, co-training
4. **Reinforcement Learning**: this involves training an agent to make a sequence of decisions by rewarding or penalizing it for the actions it takes in an environment. It's idea for sequential decision-making tasks where an agent interacts with an environment to achieve a goal. Example algorithms: Q-learning, Deep Q-Network, Policy Gradient Methods, Monte Carlo Tree Search.
5. **Self-supervised Learning**: a form of unsupervised learning where the model generates its own labels from the input data, often using a pretext task. Useful when you have large amounts of unlabeled data and  need to learn rich representations for downstream tasks. Example algorithms: autoencoding, predictive models in NLP (like BERT)
6. **Transfer Learning**: involves taking a pre-trained model (developed for a different but related task) and fine-tuning it for a specific task, where data is limited. It is idea when dealing with small datasets in domains where similar tasks have large pre-trained models (e.g. image and speech recognition). Example algorithms: fine-tuning pre-trained CNN, Transformer models in NLP (like GPT, BERT)

#### 3. What is bias and variance in ML? What is the bias-variance tradeoff?

- **Bias** refers to the error due to overly simplistic assumptions of the learning algorithm. E.g., if a linear model is used to fit a non-linear relationship, it won't capture the non-linearities, showing high bias. High bias can lead to undercutting, where the algorithm misses relevant relation between input features and target outputs.
- **Variance** refers to the error due to too much complexity in the learning algorithm, where the model caputres random noise in the data rather than the relationship between input and output. High variance can lead to overfitting, where a model learns the detail and noise in training data and cannot generalize well when making predictions on new data.
- **bias-variance tradeoff** is a fundamental issue in supervised training. Ideally we want to choose a model that appropriately balances the complexity, aka the variance, and the simplicity, aka the bias, and performs well on both training and unseen data. If a model is too simple, it may have high bias and  not fit the training data well. Conversely, if a model is too complex (too many parameters), it may have high variance and fit training data well but doesn't generalize well on unseen data. The tradeoff involes finding the right level of model complexity that achieves a low error on both training and test data.

#### 4. What are overfitting and underfitting, how can we combat them?

Overfitting and underfitting are common problems in ML, related to how well a model learns and generalizes to new data.

- **Overfitting** occurs when a model learns the training data too well, including its noise and outliers, rather than just the underlying pattern. As a result, it performs poorly on unseen data because it's overly complex.
  - Combat strategies of overfitting:
    - Simplify the model: use a simpler model with fewer parameters
    - Use more training data if you have
    - **Cross-validation**: use techniques like k-fold cross-validation to validate the model's performance on differnet subsets of the training data
    - **Regularization**: apply regularization methods (like L1 or L2 regularization) to penalize overly complex models
    - **Early stopping**: in iterative algorithms, stop training before the model fits the training data too closely
    - **Pruning (in Decision Trees)**: remove parts of the tree that provide little power to classify instances
    - **Dropout(in Neural Networks)**: Randomly omit units from the neural network during training to prevent co-adaptation of features
- **Underfitting** occurs when a model is too simple to capture the underlying pattern in the data, resulting in poor performance on both training and unseen data
  - Combat strategies of underfitting:
    - Increase model complexity: use a more complex model with more parameters 
    - Train longer
    - **Feature engineering**: add more relevant features or perform better feature selection. Reduce noise in the data.
    - Decrease Regularization

### Metrics

#### 1. Name common ML model evaluation metrics and their use cases

- Classification problems
  - Accuracy = correct/total
    - measures the ratio of correctly predicted observations to the total observations
    - limitation: can be misleading in imbalanced datasets
  - Precision and Recall
    - Use these in imbalanced data
    - **Precision** = TP/(TP + FP): out of all predicted cats, how many are really cats?
      - Use this when false positive is expensive (e.g., drug test)
    - **Recall** (**sensitivity, TPR**) = TP / (TP + FN): out of all cats, how many are caught?
      - Use this when false negative is expensive (e.g., credit card fraud)
    - **Specificity (1-FPR)** = TN / (TN + FP): out of all non-cats, how many are correctly marked as non-cat?
      - Useful when the cost of FP is high
  - F1 score = 2 * precision * recall / (precision + recall), the harmonic mean of precision and recall
    - Use this if you have imbalanced data and need a balance between precision and recall
  - ROC-AUC: Receiver Operating Characteristic - Area Under Curve
    - represents the degree or measure of separability. Tells how much the model is capable of distinguishing between classes. It is helpful in binary classification problems.
  - Confusion matrix
    - A table layout that shows TP, TN, FP, FN of a model's performance.
    - Useful for a detailed analysis of the model's performance with repsect to each class
- Regression problems
  - MAE: mean absolute error
    - The average magnitude of the errors, without considering the direction. Good for simple interpretation of a model's performance
  - MSE: mean squared error
    - The average of squares of the errors. It is more sensitive to outliers than MAE
  - RMSE: root mean squared error
    - The square root of MSE. Punishes large errors.
  - R-squared: coefficient of determination
    - represents the proportion of the variance for the dependent variable that's explained by the independent variables. Useful for comparing different regression models.

#### 2. Explain ROC, ROC-AUC and when to use it

**ROC (Receiver Operating Characteristic) curve** is a graphical representation used in binary classification to evaluate the performance of a classification model at different threshold settings. It plots two parameters: TPR (true positive rate, TP/(TP + FN)) and FPR (false positive rate, FP/(FP+TN)).

**ROC-AUC(Area Under the ROC Curve)** measures the entire two-dimensional area underneath the ROC curfe from (0,0) to (1,1). An AUC of 1 represents a perfect classifier, an AUC close to 0 means the model performs poorly, an AUC of 0.5 suggests no discriminative power (model performs the same as random).

ROC-AUC is good at dealing with imbalanced datasets. It is also useful in comparing models and chosing the optimal threshold.

### EDA & Data preprocessing

#### 1. What is EDA? What are common steps and techniques for EDA?

Exploratory Data Analysis. A critical step of data analytics that involves examining and visualizing data to uncover insights, patterns, anomalies and trends. It is crucial for understanding the data, making informed assumptions and deciding on appropriate analytical techniques and models.

Common EDA steps:

1. Summary Statistics: calculate mean, median, mode, range, variance, and standard deviation, to obtain a basic understanding of data's distribution and central tendencies.
2. Data visualization:
   1. Histogram: understand the distribution of numerical variables
   2. Box plots: spot outliers and understand the spread of data
   3. Scatter plots: identify relationships or correlations between variables
   4. Bar charts and pie charts: examine categorical data
   5. Heatmaps: visualize correlation matrix of different variables
3. Handle missing data
   1. identify and analyze patterns of missing data
   2. impute missing values: (mean, median, mode, predictive models, etc)
4. Analyze relationships and correlation
   1. Pairwise correlation analysis
   2. Cross-tabulation/Contingency tables: for catregorical data, analyze the relationship between different values
5. Identify patterns and anomalies
   1. Trend analysis: in time-series data, identify underlying patterns like seasonality, cyclic trends
   2. Outlier detection: identify data points that are signififcantly deviating from the norm
6. Feature engineering and transformation
   1. create new variables: derive new meaningful variables from existing data
   2. transform variables (log, square-root) to better expose relationships
7. Segmentation analysis: group data, segment data based on common characteristics
8. Text analysis
   1. word frequency
   2. sentiment analysis
9. Hypothesis testing: formulating and testing hypothesis

#### 2. What is data pre-processing? What are common steps and techniques of pre-processing?

Data pre-processing transforms raw data into a format that is more suitable and effective for building and training ML models. It can improve model accuracy, training efficiency, and reduce bias.

Common data preprocessing steps:

1. Data cleaning
   1. Handle missing values: 
      1. fill missing values with mean, median, mode, or using prediction models
      2. or remove rows/columns with missing values
   2. Remove duplicates
   3. Filter outliers
2. Data transformation
   1. normalization/standardization: scaling numeric data to a standard range or distribution
      1. normalization typically scale data to the range of [0,1]
      2. standardization transforms data to have normal distribution, which has a mean of 0 and standard deviation of 1
   2. Log transformation: apply logarithmic scaling to reduce data skewness
3. Data reduction
   1. Dimensionality reduction: reduce the number of features (dimensions) in the dataset using e.g. PCA, singular value decomposition (SVD), or feature selection techniques
   2. Sampling to use a subset of data when the data set is very huge
4. Categorical column encoding
   1. one-hot encoding: convert categorical variables into a form that can be provided to ML algorithms (creating dummy/indicator variables)
   2. label encoding: convert each value in a categorical column into a number
5. Handle imbalanced data: balance the class distribution
   1. Oversampling, undersampling
   2. SMOTE (synthetic minority over-sampling)
6. Text data preprocessing (for NLP)
   1. Tokenization, stemming, lemmatization: break down text into words or tokens, reducing words to their base or root form
   2. Removing stop words, punctuation, lowercasing
7. Time series data preprocessing
   1. Handle seasonality and trends
   2. Lag features, rolling window statistics: create features that capture temporal dependencies
8. Feature engineering
   1. create new features
   2. feature selection

#### 3. How to handle imbalanced data?

1. Resampling
   1. oversampling the minority class
      1. Simply duplicating instances of the minority class, or
      2. Use more sophiscated techniques like **SMOTE**(Synthetic Minority Oversampling Technique) to generate synthetic samples
   2. Undersampling the majority class
      1. Randomly reduce instances of the majority class, or
      2. Use advanced techniques that try to retain infgormation, like **Tomek links** or **Cluster Centroids**
2. Modify class weight
   1. assign a higher weight to the minority class, you can increase the cost of misclassifying these instances and promote the model to pay more attention to them
3. Use different evaluation metrics
   1. traditional metric like **accuracy** is misleading. Use metrics like **precision**, **recall**, **f1 score**, or **AUC-ROC** which provide more insight into model performance on imbalanced dataset
4. Anomaly detection
   1. Sometimes, treating the problem as an anomaly detection problem can be effective, especially when the minority class is very few
5. Use algorithms that are robust to imbalance
   1. e.g., Tree-based methods are naturally more robust to class imbalancre
6. Ensemble methods
   1. Bagging and boosting can be effective. Boosting adjusts the weight of an instance based on the last classification and can therefore increase the weight of misclassified minority instances

#### 4. What is feature engineering and how do you do it?

Feature engineering is essentially the process of using domain knowledge to extract features from raw data. It can involve creating new feautres, reducing irrelevant features, or modifying existing features. It improves model performance.

- Feature creation
  - Combine: combine income and debt features into a debt-to-income ratio feature
  - Decompose: extract hour, day, weekday, month from timestamp feature
  - Transform: apply a log transformation to a highly skewed feature like "income" in a dataset
  - Aggregate: for stock price, use rolling averages like 7-day moving average as a new feature
- Feature encoding
  - One-hot encoding: convert categorical feature into separate features with binary values (color: red, blue, green -> color_red: 0/1, color_blue: 0/1, color_green: 0/1)
  - label encoding: convert label in to numerical values
- Feature scaling
  - scale numerical features to the same range (e.g., 0, 1) so that a feature with bigger range (like salary) doesn't get too much weight and skew the model
- Feature selection
  - Filter: use a correlation matrix to remove features that are highly correlated
  - Wrapper: use recursive feature elimination in logistic regression to identify which features contribute most to predicting
  - Embedded: use a random forest model, which inherently performs feature selection, to identify important features for predicting customer churn
- Dimensionality reduction
  - E.g., when you have hundreds of features in a facial recognition task, apply PCA to reduce the number of features while retaining the essence of the data

Do feature engineering iteratively, assess the model's performance and adjust. The key is to understand the data and the underlying problem well, and creatively apply domain knowledge and statistical techniques to engineer features that help the model learn better.

### Training

#### 1. What is training testing valdidation data split? How much data to allocate to each training/testing/validation set?

Spliting data into training, testing, validation sets is a fundamental practice to evaluate the performance of models. (Shuffle/randomize your data before splitting)

- Training set: used to train the model. Typically 60-80% of all data
- Validation set: used to evaluate the model fit during training. Crucial for tuning parameters and preventing overfitting. Typically 10-20%
- Testing set: used to evaluate the final model after the training is complete. Typically 10%-20%.

Splitting strategies:

- Holdout
- K-fold cross-validation: data is divided into k subsets, the holdout method is repeated k times, each time, one of the k subsets is used as the test set and the other k-1 subsets are put together to form a training set. 
- Stratified splitting: maintain the distribution in subsets

#### 2. What is ensemble learning? What is bagging, boosting, stacking and when to use which?

Ensemble learning is a ML paradigm where multiple models (weak learners) are trained to solve the same problem and combined to get better results. Bagging, boosting, and stacking are three common ensemble learning techniques

- **Bagging** (Bootstrap Aggregating): multiple models are trained in parallel, each on a random subset of the data. The subset are created by bootstrap sampling, which means sampling with replacement. The final prediction is typically the average of all predictions for regression, or the majority vote for classification.
  - **Random Forest** is a classic example of bagging
  - Use bagging when your model is overfitting or has high variance. Effective for complex models like decision trees
- **Boosting**: train weak models in sequence, each model tries to correct the errors made by previous one (assigning more weight to the wrongly predicted samples). The final prediction is a weighted sum of the predictions made by the individual models.
  - **AdaBoost** (Adaptive Boosting) and **Gradient Boosting** are popular boosting algorithms
  - Use boosting when your model is underfitting, or has high bias. Useful when the dataset is balanced and not too large.
- **Stacking**: train multiple models (usually of different types) and then train a meta-learner to combine their predictions. The base level models are trained on full dataset and then the meta-model is trained on the outputs of the base models as features
  - Use stacking when you want to blend the strengts of various individual models. Useful when you have a complex problem and wish to combine the predictions of several models to improve accuracy

#### 3. What is regularization? Name and explain some regularization techniques

Regulairzation is a technique to prevent overfitting by penalizing models for their complexity. Overfitting happens when a model learns not only the underlying patterns in the training data but also the noise, resulting in poor performance on unseen data. Regularization adds a penalty term to the objective function that the algorithm is trying to minimize, encouraging simpler models during training.

Common regularization techniques:

1. **L1 Regularization (Lasso Regression)**: adds a penalty equal to the absolute value of the magnitude of coefficients (the sum of the absolute value of coefficients) to the loss function
   1. This can lead to some coefficients being shrunk to zero, making the data sparse, effectively performing feature selection. 
   2. Useful when you suspect some features are not important and want the model to reflect that
2. **L2 Regularization (Ridge Regression)**: adds a penalty equal to the square of the magnitude of coeeficients (the sum of the square of coefficients) to the loss function
   1. Doesn't result in 0 coefficients, but encourages them to be small. Effective at handling collinearity (high correlation between variables)
   2. Commonly used when you have a dataset with many features
3. **Dropout (in NN)**: during training, randomly ignore some neurons' outputs of a layer, turning them off for downstream neurons.
   1. It reduces neuron co-adaptation, where neurons rely too heavily on the presence of specific other neurons during training and cannot operate independently, reducing the network's generalizing ability.
   2. Common in NN, especially when training large NN
4. **Early Stopping**: stop training before the learner passes a certain number of iterations or achieves a certain accuracy on the validation set.
   1. Prevents the model from learning the noise in the training data
   2. Applicatble in many training algorithms, especially in NN where training can be stopped once the validation error starts to increase

### Algorithms

#### 1. What are some supervised learning and unsupervised learing algorithms and their applications?

- Supervised learning algorithms
  - Linear regression: predicts a continuous outcome, used in predicting prices, sales, life expectancy
  - Logistic regression: binary classification. Email spam detection, loan approval, medical diagnosis
  - Decision Tree: a tree-like model of decisions and their possible consequences. Customer segmentation, fraud detection
  - Random Forest: an ensemble of decision trees, typically trained with bagging. Stock market prediction, e-commerce recommendation system
  - Support Vector Machine (SVM): finds a hyperplane in N-dimensional space that distinctly classifies data points. Face detection, text and hypertext categorization, image classification
  - Neural Networks: speech recognition, image recognition, financial time series prediction
- Unsupervised learning algorithms
  - K-menas clustering: partitions data into clusters based on feature similarities. Market segmentation, data compression, image segmentation
  - PCA: reduces dimensionality by finding new features that maximize variance. Feature extraction and data visualization, noise reduction
  - Autoencoder: NN that learns to encode input data into a few key features and then decod it back. Data denoting, dimensionality reduction, anomaly detection

#### 2. When will you use classification, when regression? Name some common algorithms for each

Use classification when the target variable is categorical (discrete) and the task is identify which category or class the observation belongs to. Use regression when the target variable is continuous or numerical, and the task is predicting a quantity. 

- Classification algorithms and use cases:
  - Logistic Regression: used in binary classification, like spam filter
  - SVM (Support Vector Machines): used in image classification and bioinformatics, where clear margins of separation are important
  - KNN (K nearest neighbors): common in recommendation systems, where the algorithm finds similar items
  - Decision Tree: loan approval
  - Random Forest: medical diagnosis where multiple decision trees increase prediction accuracy
- Regression algorithms and use cases
  - Linear Regression: predict housing prices, sales forcasting
  - Polynomial Regression: where relationship between variables is not linear, like economic growth prediction
  - Ridge Regression: when there's multicollinearity in the data, like in multicriteria decision-making
  - Lasso Regression: useful in feature selection, as it tends to shrink coefficients of less important features to zero
  - Elastic Net Rergession: combines lasso and ridge regression, used in scenarios where you need to balance feature selection and multicollinearity, like quantitative genetics
  - Decision Tree and Random Forest for Rergession: complex regression tasks with non-linear relationships, like predict energy consumption of a building

#### 3. How do you choose which ML algorithm to use?



#### 4. What is PCA? When and how do you use it?

PCA is a method to reduce dimensionality of data by finding new, fewer variables (principal components) that capture the essential variance (information) in the original variables. 

You can use PCA when you have a dataset with many interrelated variables and you want to simplify the dataset, while still retaining as much information as possible. It is particularly useful in visualizeing high-dimensional data, reducing computational burdens, and removing noise from data.

To apply PCA:

1. Standardize your data: scale each feature so that it has mean of zero and standard deviation of one. This ensures that all features contribute equally to the analysis and the results are not skewed by the original scale of the data
2. Compute covariance matrix: the covariance matrix expresses how each variable in the dataset relates to every other variable. 
3. Calculate eigenvalues and eigenvectors of the covariance matrix. **Eigenvectors** are directions in the feature space along which the data varies the most, **eigenvalues** are values that indicate the magnitude of this variance.
4. Choose principal components: the eigenvetors with the highest eigenvalues are the principal components. They represent the directions of maximum variance.
5. Project data onto Principal Components: project the original data onto these principal components and we have a new dataset with reduced dimensions where the features are uncorrelated and most of the information in the original dataset is retained.



## Deep Learning

### Basics

#### 1. What is Deep Learning and how is it different from machine learning?



#### 2. What is a neural network? What is a neuron?



#### 3. What are the layers of a typical NN? What are their uses?



### Training

#### 1. What is gradient descent?



#### 2. What is back propogation?



#### 3. What is a loss function? Name common loss functions and their use cases



#### 4. What is an activation function? Why do you use it? Explain sigmoid, ReLU, leaky ReLU, tanh, softmax and when to use them



#### 5. What is weight and bias in a NN? How do you initialize weights?



#### 6. What are optimizers? Name some optimizers and explain when to use which



#### 7. What are hyperparameters? How do you tune hyperparameters?



#### 8. What is SGD? When and how to use it?



#### 9. What is mini-batch? When and how to use it?



#### 10. What is MLP? How do you implement it? When and how do you use it?



#### 11. What is vanishing gradient and exploding gradient? How to combat them?





### Algorithms

#### 1. Name common Deep Learning algorithms and their use cases



#### 2. Name NLP algorithms and how they work



#### 3. Name CV algorithms and how they work



### Evaluation

#### 1. What are common evaluation metrics for Deep Learning? When to use which and why?



## Generative AI & LLM

