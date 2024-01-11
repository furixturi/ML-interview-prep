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
   1. This can lead to some coefficients being shrunk to zero, making the data sparse, <u>effectively performing feature selection.</u> 
   2. Useful when you suspect some features are not important and want the model to reflect that
2. **L2 Regularization (Ridge Regression)**: adds a penalty equal to the square of the magnitude of coeeficients (the sum of the square of coefficients) to the loss function
   1. Doesn't result in 0 coefficients, but encourages them to be small. Effective at handling collinearity (high correlation between variables)
   2. <u>Commonly used when you have a dataset with many features</u>
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

1. Type of problem: classification? Regression? Clustering?
2. Size and quality of data at hand
   1. Small dataset: deep learning require large amounts data. Also you're at more risk of overfitting. Consider algorithms with buil-in regularization like **Ridge (L2) or Lasso (L1) regression**, or ensemble methods like **Random Forest**
   2. quality of data: if your dataset has lots of missing values or noise, start with algorithms that are robust to this, e.g., **Random Forest**
3. Computation time
   1. consider training time and prediction time you are aiming for. Models like SVM or DL are computationally intensive and needs a lot of computation resource and time
4. Interpretability
   1. Industries like health and finance have high compliance requirements and may prefer simpler, more interpretable models such as Decision Trees, Logistic Rergession

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

DL is a subset of ML. It involves algorithms inspired by the structure and function of brain, called artificial neural networks. The "deep" refers to that there are multiple layers in such networks, through which data is transformed and complex pattern is learned. 

DL vs ML:

- DL can handle large amount of complex datasets, like images, sound, and text, where relationships between input and output are non-linear and not easy to interpret. Thus DL is well suited for complex problems such as image recognition, NLP, and speech recognition, while traditional ML is usually better dealing with tabular data and can be more efficient for simpler tasks. 
- DL requires a larger amount of training data than traditional ML algorithms.
- DL is more computaionally intensive due to its multiple layers and more complex model architecutre.
- DL can reduce the need for feature engineering, as it automatically detects relevant features. Traditional ML often relies heavily on feature engineering
- Traditional ML models are more easily to interpret than DL models

#### 2. What is a neural network? What is a neuron?

A **Neural Network** is a computational model inspired by the way biological neural networks in human brain process information. A NN is composed of layers of interconnected units or nodes called "neurons". These typically include an input layer, one or more hidden layers, and an output layer. Each neuron receives input from the previous layer, processes the input, and passes its output to the next layer. The network learns by adjusting the weights of connections between neurons. 

A **Neuron** is a basic union of neural network. It receives one or more **inputs**, which can be features from the dataset or output from other neurons. Each input has an associated **weight**, which is a trainable parameter that the network learns during training. The neuron applies an **activation function** to the weighted sum of its input, which calculates whether or to what extend the signal should be further propagated through the network.The result of the activation function is the **ouput** of the neuron, which then becomes the input to the neuron in the next layer.

#### 3. What are the layers of a typical NN? What are their uses?

A typical NN would have several layers, including the input layer, one or more hidden layers, and an output layer.

**Input Layer** receives the raw input data. The number of neurons in this layer usually correspond to the number of features in the dataset.

**Hidden Layers** perform computations on the inputs received from previous layer, using **weights**, **biases**, and **activation functions**. The complexity and depth of a NN are determined by the number and size of these hidden layers. Different layers can learn different aspects of the data. Early layers might learn basic features, while deeper layers can learn more abstract concepts.

**Output Layer** produces the final output. For classification, it often uses a softmax activation function to output a probability distribution over classes (or sigmoid for binary classification.) For regression tasks, it might have a single neuron that outputs a continuous value.

**Common hidden layers** include

- **Dense (Fully Connected) Layers**: each neuron receives input from all neurons of the previous layer, making them fully connected. This is used in most types of NN
- **Convolutional Layers**: they apply a convolution operation to the input. which are commonly used in CNNs for image processing tasks
- **Pooling Layers**: also used in CNNs, usually following convolutional layers. They serve to reduce the spatial dimensions (width and height) of the input volume for the next convolutional layer
- **Recurrent Layers**: used in RNNs for processing sequential data such as time series and text. They have loops to persist information from one step of the sequence to the next, maintaining a "memory" of previous input.
- **Normalization Layers**: used to normalize the output of the previous layer, improving the stability and speed of the network training.
- **Dropout Layers**: used to prevent overfitting by randomly dropping out a subset of neurons and their connections during training.

### Training

#### 1. What is gradient descent?

Gradient descent is an optimization algorithm used in ML to minimize cost function, update parameters (weights) to find the minimum cost function (the cost function measures the difference between the model's prediction and the actual data).

At each step, the gradient (or the slope of the function, which points to the steepest ascent) is calculated, which is the partial derivatives of cost function with respect to each parameter (weight). Then to minimize the function, the parameters are updated by moving them to the opposite direction of the gradient. The process is iterated until the algorithm converges to the minimum of the function.

How much to update at each step is decided by the hyper parameter **learning rate**. A small learning rate make the learning process slow and might stuck at local optimum point, while a too large learning rate might cause overshooting and never converge to the minimum. 

Different types of gradient descent:

- **Batch Gradient Descent**: compute the gradient using the whole dataset. This can be very slow and computationally expensive for large datasets
- **Stochastic Gradient Descent (SGD)**: compute the gradient and update the parameters one by one for each data point. It's faster but has a higher variance in the parameter updates, which can cause the cost function to fluctuate.
- **Mini-batch Gradient Descent**: a compromise between batch and stochastic gradient descent. It computes the gradient and updates the parameters on small batches of data. This is most commonly used.

#### 2. What is back propagation?

Back propagation is an algorithm used in training NN, especially DL. It is a mechanism used to update the weights of the network efficiently in order to reduce the network's prediction error.

How it works:

1. First go through the **Forward pass**, where data is passed through the network, from input layer through each layer using the current set of weights, to compute the output.
2. Then we calculate the loss with the output, which measures how for the network's output is from the actual target value
3. Now we enter the **Backward pass**. The loss is propagated back through the network in reverse, from output layer back to the input layer. During this pass, the partial derivatives with respect to each weight is calculated. This involves applying the chain rule of calculus to compute the gradient of the loss function with respect to each weight.
4. The weights are then updated with the calculated gradient. The size of the update is determined by a hyperparameter called learning rate.
5. Repeat this process for many iterations or epochs over the entire dataset, until the network weights converge to a state where the loss is minimized. 

#### 3. What is a loss function? Name common loss functions and how do we choose them

A loss function, also called cost function or error function, quantifies the difference between the predicted values and the actual values in the data. The goal of ML training is typically to minimize the loss function.

Common loss functions:

- **MSE (Mean Squared Error)**: calculates the average of the squares of the errors. Used in regression tasks. Good if your dataset doesn't have outliers and you want to punish larger errors.
- **MAE (Mean Absolute Error)**: calculates the average of absolute differences between predicted values and actual values. Used in regression tasks. Good if your data contain outliers or you want to treat all errors equally.
- **Cross-Entropy (Log Loss)**: measures the performance of a classification model whose ouput is a probability value between 0 and 1. Used in classification tasks.
- **Hinge loss**: used for SVM.
- **Huber loss**: combines MSE and MAE, the loss is quadratic for small errors and linear for large errors, making it robust to outliers in regression tasks.

#### 4. What is an activation function? Why do you use it? Explain sigmoid, ReLU, leaky ReLU, tanh, softmax and when to use them

Activation function in NN is a mathematical function applied to the output of a neuron. It introduces non-linearity to the NN, allowing it to learn and perform more complex tasks. They also help to normalize the output of each neuron, e.g., between 0 and 1 or -1 and 1.

Common activation functions:

- **Sigmoid (Logistic Function)**: squashes the input value between 0 and 1. Often used in binary classification's output layer. Limitation: can cause vanishing gradient problem. not zero-centered
- **ReLU (Rectified Linear Unit)**: outputs the input if it is positive, 0 otherwise. Used in hidden layers. It helps overcome the vanishing gradient problem and allows faster training. Limitation: it can cause neurons to die during training.
- **Leaky ReLU**: similar to ReLU but allows a small, non-zero gradient when the unit is not active (Leaky ReLU(x) = max(0.01x, x)). Can be used in hidden layers where ReLU is applicable. It might fix the dying neuron problem of ReLU.
- **TanH (Hyperbolic Tangent)**: squashes values between -1 and 1. Used in hidden layers. It's zero-centered which makes optimization easier in some cases. Limitations: it can still suffer from vanishing gradient problem
- **Softmax**: converts a vetor of numbers into a vector of probabilities that sum up to 1. Commonly used in ouput layer of multi-class classifier.

#### 5. What is weight and bias in a NN? How do you initialize weights?

Weights in a NN are the parameters that multiply with the input data. Each connection between neurons in different layers has an associated weight. These weights are adjustied during training to minimize loss. The collection of all weights essentially defines the learned model.

Bias is an additional parameter added to the weighted sum of inputs before passing it through an activation function. It acts like an intercept term in a linear equation. It allow the network to shift the activation to the left or right, rather than always passing through the origin of the input space.

Ways to initialize weights:

- **Random initialization**: initialize weights to random small numbers. Initializing them too large or too small can cause exploding or vanishing gradient problem
- **Xavier/Glorot Initilaization**: initialize the weights with values from a distribution with zero mean and variance 1/(number of input). Good for using tanh activation.
- **He initialization**: initialize the weights with values from a distribution with 0 mean and variance 2/(number of inputs). Good for using ReLU
- **Orthogonal initialization**: initialize the weights as orthogonal matrices. Beneficial for deep networks since it helps preserve the magnitude of backpropagated gradients and can reduce risk of vanishing/exploding gradient.

#### 6. What are optimizers? Name some optimizers and explain when to use which?

Optimizers are algorithms and methods that help train a NN.

Common optimizers:

- **Gradient descent**: the most basic optimizer. It updates the weights by directly subtracting the gradient of loss function with respect to the weights. Not used much due to its inefficiency.
- **SGD**: a variation of gradient descent. SGD updates the weights using only a single data point. It's more efficient than standard gradient descent and is widely used.
- **Momentum**: helps accelerate SGD in the relevant direction and dampens oscillations. It does this by adding a fraction of the update vector of the past step to the current update vector. It's useful for faster convergence and dealing with the ravines or plateaus in the loss landscape.
- **Adagrad**: adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. Good for sparse data.
- **RMSprop**: address some of Adagrad's issue by using a moving average of squared gradients to normalize the gradient, leading to better performance in the online and non-stationary settings.
- **Adam (Adaptive Moment Estimation)**: combines Momentum and RMSprop. It computes adaptive learning rates for each parameter. Fairly robust and widely used.
- **AdamW**: a variation of Adam with a different way of handling weight decay. It decouples the weight decay from the gradient updates, which can lead to better performances in some cases.

How to choose optimizer:

- Simple, convex problems or problems with a small amount of data: basic SGD
- Deep learning tasks: Adam is often a good starting point
- Where precision is crucial (like training a language model): RMSprop or Adagrad, since adjusting learning rate based on the parameters can be beneficial
- Very deep NN or NN with a complex structure: AdamW or SGD with momentum might provide better convergence

#### 7. What are hyperparameters? Name some hyperparameters and explain how you choose their initial values. How do you tune hyperparameters?

Hyperparameters are parameters that set prior to the training process and are not learned from the data. 

Some common hyperparameters:

- **Learning Rate**: determines the step size at each iteration to adjust weight towards minimum loss. Too large can cause model to overshoot and never converge, too small can make the training process needlessly long and complex. Initial vvalue is often set to a small value like 0.001 or 0.01, or use adaptive rates like Adam optimizer.
- **Number of epochs**: the number of times the training process to walk through the entire dataset. Too few can lead to underfitting, to many can lead to overfitting.
- **Batch size**: the number of training examples used in one iteration. Smaller batch size typically mean more updates in one epoch, which requires more computation but can lead to faster convergence. Common batch size choices are 32, 64, 128, depending on available memory
- **Number of layers and neurons in each layer**: defines the NN architecture. More layers and neurons make the model more complex, which can handle more complex problems but can also lead to overfitting 
- **Activation functions**: like ReLU, Sigmoid, Tanh
- **Regularization parameters**: like L1 or L2 regularization, dropout, etc. to prevent overfitting

Ways to tune hyperparameters:

- **Grid Search**: test every combination of a predefined list of values. It's exhaustive but time cosuming.
- **Random Search**: Randomly selecting combinations of hyperparameter values. This can be more efficient than grid search.
- **Bayesian Optimization**: use a probablistic model to predict the performance of hyperparameters and choose new hyperparameters based on this model 
- **Automated Hyperparameter Tuning Tools**

#### 10. What is MLP? How do you implement it? When and how do you use it?

**MLP (Multi Layer Perceptron)** is a foundational architecture in neural network. It consists of an input layer, one or more fully connected hidden layers, and an output layer. Activation functions like ReLU, sigmoid or tanh are used to introduce non-linearity. Training MLP uses back propagation. 

MLP is typically used for supervised learning problems. It can be used in various simple to moderate complexity tasks, especially where the data doesn't have a significant spatial or temporal component. For more complex tasks involving images, videos, or sequential data, more specialized architectures like CNN or RNN are usually more effective.

#### 11. What is vanishing gradient and exploding gradient? How to combat them?

Vanishing gradient and exploding gradient are two common problems encountered when training deep neural networks, especially with traditional activation functions like sigmoid or tanh. They are primarily related to how gradients are propagated back through the network during training, where gradients are multiplied by weights and derivatives of the activation functions.

**Vanishing gradient** occurs when the gradients of the network's weights become very small, effectively preventing the weights from changing their values. In deep network, gradients are back-propagated from output to input layer, during which they get multiplied by the weights and the derivatives of the activation functions. If these numbers are small (<1), the gradient can diminish exponentially as they reach early layers, making it very hard for the network to learn and converge especially for early layers.

Mitigation methods:

- Use **ReLU or Leaky ReLU** as activation functions. Positive inputs will have their gradient to be 1
- **Use Shorter Networks or Residual Connections**: use networks with fewer layers, or use "Residual Connections" like in ResNet (which allow gradients to bypass certain layers throught the addition of a shortcut connection) can help alleviate vanishing gradient problem

**Exploding gradient** is the opposite of vanishing gradient. It occurs when the gradients of the network's weight become excessively large. This can cause the weights to oscillate or diverge, rather than converge, during training. It is also a result of back-propagation process, but in this case, the gradients grow exponentially through the layers due to large weights or derivatives.

Mitigation methods:

- **Gradient clipping**: this is to prevent exploding gradient especially in RNNs. It scales down the gradient if they exceed a set threshold, ensuring they don't grow too large.

To combat vanishing and exploding gradient:

- **Use ReLU or Leaky ReLU Activation Functions** to help mitigate vanishing gradient problem.
- **Weight initialization**: use He or Xavier initialization can set the weights to optimal values based on the number of input and output neurons.
- **Batch normalization**: this normalizes the input layer by adjusting and scaling the activations. It can mitigates the problem by maintaining the mean ouput close to 0 and the output standard deviation close to 1.
- **Gradient clipping**: this is to prevent exploding gradient especially in RNNs. It scales down the gradient if they exceed a set threshold, ensuring they don't grow too large.
- **Use Shorter Networks or Residual Connections**: use networks with fewer layers, or use "Residual Connections" like in ResNet (which allow gradients to bypass certain layers throught the addition of a shortcut connection) can help alleviate vanishing gradient problem
- **Skip Connections and Highway Networks**: these structures allow gradients to flow across several layers without undergoing too much transformation
- **Use LSTM/GRU for RNNs**: in the case of recurrent neural networks, using LSTM (Long Short Term Memory) or GRU (Gated Recurrent Units) can help avoid these issues



### Algorithms

#### 1. Name common Deep Learning algorithms and their use cases

- **CNN (Convolutional Neural Networks)**: a type of DL algorithm designed for processing data that has grid-like topology, such as images. Its layers apply convolutional filters that capture spatial hierachies and features like edges, textures, and more complex patterns in deeper layers. 
  - Use cases: CNNs are widely used in CV tasks such as image analysis, object detection, facial recognition, autonomous vehicle perception systems
  - Key components: 
    - **Convolutional layers** are the core building blocks of a CNN. They perform a mathematical operation called convolution, which involves sliding filters (or kernel) over the input data (the image) to produce feature maps. The filters are applied over the intire image and are used to detect specific features like edges, shapes, contextures.
    - **Activation function**: after each convolution, an activation function (like ReLU) is applied to introduce non-linearity, allowing the network to solve more complex problems.
    - **Pooling layers**: these layers reduce the spatial size (width and height) of the input volume for the next convolutional layer. Max pooling is most commonly used, where the maximum element is selected from the region of the feature map covered by the filter. Pooling helps reduce the number of parameters and thus reduce the computation. It also controls overfitting.
    - **Fully connected layers**: after several convolutional and pooling layers, the final output is flattened and fed to a fully connected layer, where high-level reasoning such as classification is done.
  - Limitations: CNNs require large quantity of labeled training datasets, and they can be computationally intensive. CNN is also prone to overfitting due to its deep architecture, especially when there's not much training data.

- **GAN (Generative Adversarial Networks)**: famous for creating highly realistic images. GANs consist of two parts: a generator that creates samples and a discriminator that evaluates them. The generator learns to produce more and more realistic data, while the discriminator gets better at distinguishing real data from fake. This adversarial process improves both networks.
  - Use cases: video generation, creating virtual environments
  - Limitations:  difficult to train due to issues like mode collapse, where the generator produces limited varieties of output and non-convergence. They require large datasets and significant GPU resources.
- **Seq2Seq (Sequence to Sequence)**: a type of models that primarily used to map an input sequences to ouput sequences, like translation. They consist of an encoder and a decoder, both typically RNNs or LSTMs. The encoder processes the input sequence, and the decoder generates the output sequence.
  - Use case: machine translation, text summarization, speech recognition
  - Limitation: 
    - struggle with long sequences handling. The longer the sequence, the more challenging to retain all relevant information untill the end, leading to issues like vanishing gradient.
    - sequential processing can lead to inefficiencies in training and inference, unlike Transformers, seq2seq can't process data in parallel.
- **RNN(Recurrent Neural Network)**
- **LSTM (Long Shrot-Term Menory Networks)**

### NLP

#### 1. Name common NLP tasks

- named entity recognition (NER): identify key entities in text into predefined categories such as names of persons, organizations, locations.
- machine translation
- sentiment analysis
- topic modeling
- speech recognition
- summarization
- question answering
- text generation

#### 2. What is TF-IDF?

Term frequency-inverse document frequency (TF-IDF) is a classical text representation technique in NLP. It uses a statistical measure to evaluate the importance of a word in a document relative to a corpus of documents. It is the combination of two terms: term frequency(TF), inverse document frequency(IDF).

- Term Frequency (TF): how frequently a word appears in a document. A higher TF indicates that a term is more important in that specific document.
- Inverse Document Freauency (IDF): measures how rare or unique a term is across documents in the entire corpus. It down weight the terms that frequently appear in all documents and update the importance of the rare terms.

TF-IDF score is calculated by multiplying the TF and IDF for each term in a document. The result indicates the term's importance in the document and corpus. Terms that are frequent in a document but uncommon in the corpus will have high TF-IDF scores, suggesting their importance in that document.

#### 3. What are common pre-processing techniques in NLP?

NLP requires preprocessingthe raw text input to clean and change text data so that it may be processed or analyzed. The preprocessing typically involves a series of steps, which can include:

- Tokenization
- Stop-Word Removal
- Text Normalization
  - Lowercasing
  - Lemmatization: convert words to their base or dictionary form, known as lemmas. E.g., "running" to "run", "better" to "good"
  - Stemming: reduce words to their root form by removing suffixes or prefixes. E.g., ""playing" to "play", "cats" to "cat"
  - Abbreviation expansion
  - numerical normalization: convert numerical digits to written form 
  - Date and time normalization
- Remove special characters and punctuation
- Remove HTML tags or markup
- Spell correction
- Sentence segmentation

#### 4. What is tokenization in NLP?

Tokenization is the process of breaking down text or string into smaller units called tokens. These tokens can be words, subwords, or characters. It is a fundamental step in NLP tasks. 

#### 5. What is stemming and lemmatization? How are they different from each other?

Stemming and lemmatization are two comonly used word normalization techniques in NLP, which aim to reduce the words into their base or root forms. Their goals are similar but approaces are different.

- Stemming typically chopps off the word's suffixes or prefixes using heuristic or pattern-based rules, regardless of the context. E.g., cats to cat. It is simple but sometimes doesn't produce actual words, e.g., universe and university might both be stemmed to "univers"
- Lemmatization reduces the word to its dictionary form, called lemma. It uses linguistic knowledge, taking into account the word's context, tense, number, etc. The output lemma are valid per dictionary. E.g., running, ran, runner would all result in "run", "better", "best", will be ouput to "good". Lemmatization can be more accurate than lemmatization but more complex and computationally intensive. 

#### 6. What are embeddings?

Word embedding in NLP is a technique to represent text in a form that computer can process effectively. It convert words, phrases into numerical vectors in a high-dimensional vector space. These vectors capture semantic and syntactical features of the text, allowing words with similar meanings or other relationships to be represented closely in the vector space and thus allow algorithms to understand and process language effectively. It is also more efficient than the traditional text representation methods like one-hot encoding, as the latter produces high-dimensional sparse vectors that takes a lot of memory space.

Common algorithms to create word embeddings:

- Word2vec: uses Continuous Bag of Words (CBOW) to predict a word based on its context, and Skip-gram, which predicts the context given a word.
- GloVe (Global Vectors for Word Representation): creates word vectors by analyzing word co-occurences in a corpus
- BERT: consider the context of each word in both directions, uses transformer architecture, is context aware

#### 7. RNN

An NLP algorithm. 

RNNs process sequential data by maintaining a hidden state that captures information about previous elements in the sequence. This allow them to make predictions based on both the current input and the context provided by oreceding sequence.

Use cases: text generation, sentiment analysis, language modeling

Limitations: difficulty handling long sequences due to vanishing gradient problem. Slow training and inference due to sequential processing

#### 7. LSTM

A type of RNN that is designed to address RNN's problem of learning long range dependences in sequence data. It has memory cells that can maintain information over extended periods. These cells are equipped with special units called gates - input gate, forget gate, output gate, to control the flow of information. They can retain important information over long sequences and forget irrelevant information.

Use cases: machine translation, speech recognition, text summarization

Limitations: computaionally intensive, require large datasets, and can be slow to train.

#### 8. GRU

A type of RNN that is designed to efficiently capture dependencies in sequencial data, similar to LSTM but has a simpler architecture. It has two gates, update gate and reset gate to control information flow. It doesn't have separate memory cells. 

GRU is more memory efficient due to its simpler structure, making it a better choice for smaller datasets or use cases where computational efficiency is a priority.

### Evaluation

#### 1. What are common evaluation metrics for NLP? When to use which and why?



## Generative AI & LLM

### Transformer

#### 1. What is the Transformer model?



#### 2. Describe the Transformer model architecture



#### 3. What is attention mechanism in Transformer model?



#### 4. What is self-attention mechanism?



#### 5. What is the purpose of the multi-head attention mechanism?

