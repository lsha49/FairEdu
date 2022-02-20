# Towards fairer education for all
An python repository to perform educational dataset balancing applied in submitted paper in @todo. 
Download this repository with `git clone` or equivalent.

```bash
git clone @repo
```

## Requirements  
* Python 3.8  
* Tensorflow > 1.5
* tensorflow-estimator 2.7.0
* tensorflow-macos 2.7.0
* tensorflow-metal 0.3.0
* Sklearn > 0.19.0  


## Datasets & Model implementation
We describe below the detailed dataset and model implementation.

### Forum Post
The forum dataset is included as [Moodle Forum de-identified Embeddings](https://github.com/). The CNN-LSTM model was implemented using a tensorflow repo modified from [here](https://github.com/zackhy/TextClassification), and the BERT embeddings of input text were generated using [Bert-as-service](https://github.com/hanxiao/bert-as-service). We set the input layer dimension as 768 with a sigmoid output layer. The L2 regularizer lambda set to 0.001. For the CNN network, 128 convolution filters with a width of 5 was used. For the LSTM network, 128 hidden states and 128 cell states was used. During training, one cycle policy was used with the batch size of 32, training epochs of 50, maximum learning rate of 2e-05 and dropout probability of 0.5. The shuffling was performed at the end of each epoch and with an early stopping mechanism. After each epoch, 10% of the training data was chosen at random as validation data and the best model was selected based on the validation error. 


### Legal Assignment
The Caseote is included as [Legal Casenotes de-identified Embeddings](https://github.com/). we adopted the open-sourced huggingface[Legal BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased) model. We fine-tune BERT Legal by a broader hyper-parameter search space procedure proposed in Legal BERT, with batch size set to 8 without a fixed maximum epochs to avoid model under-fitting and apply early stopping based on validation loss. We adopted the same setting of low learning and high drop-out rate of 1e-5 and 0.2 respectively as it was shown to improve regularization.

### KDDCUP Student Dropout 
The KDDCUP dataset is linked in [KDDCUP](https://data-mining.philippe-fournier-viger.com/the-kddcup-2015-dataset-download-link/). We implemeneted [this](https://github.com/wzfhaha/dropout_prediction) model by Feng. In line with the original implementation, the model had 32*32 deep layers with dropout ratio set to 0.5 and 32 convolution filters with a width of 8. During training, the learning rate was set to 0.001 with Adam optimizer, 10 epoch, 256 batch and logloss type. After each epoch, 10% of the training data was chosen at random to serve as validation data and the best model was selected based on validation error.

### Open University Student Performance
The Open University dataste is linked in [Open University](https://analyse.kmi.open.ac.uk/open_dataset#description). The source code is provided in [OUA](https://github.com/gogoladzetedo/Open_University_Analytics). For the traditional ML model, we applied the GridSearchCV package to find the best model hyperparameters to optimise the performance. 
```python
A sample GridSearched model: 
rfc=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    max_depth=None, max_features=0.5, max_leaf_nodes=None,
    min_impurity_split=1e-07, min_samples_leaf=1,
    min_samples_split=4, min_weight_fraction_leaf=0.0,
    n_estimators=250, n_jobs=1, oob_score=False, random_state=None,
    verbose=0, warm_start=False)
```

### ADM STEM career
The ADM dataset is linked in [ADM STEM career prediction](https://sites.google.com/view/assistmentsdatamining/data-mining-competition-2017?authuser=0). We implemented this [model](https://github.com/ckyeungac/ADM2017). For the traditionla ML model, we applied the GridSearchCV package to find the best model hyperparameters to optimise the performance. 
```python
A sample GridSearched model: 
lrc = LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
    fit_intercept=True, intercept_scaling=1, max_iter=100,
    n_jobs=1, penalty='l1', random_state=None,
    solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
```


## Dataset Balancing implementation
The dataset balancing is implemented by ```python cbt(self, X, Y, G):```, where X is input features, Y is input label and G is input demographics. Then to re-sample with the traditional Class Balancing Strategy, simply apply SMOTE from ```python imblearn.over_sampling ```  package: ```python SMOTE().fit_resample(X, Y) ```, or any other class balancing techqniues such as BorderlineSMOTE, NearMiss. To re-sample with demographics, simply add demographic into class parameter and treating the balancing as a multi-class balancing problem, i.e.,  ```python GY = G.astype(str) + Y.astype(str) ``` and then ```python X, GY = SMOTE().fit_resample(X, GY) ```.

## Hardness Bias 
After generating samples, we evaluate the kDN distribution by ```python distance.jensenshannon ``` and selected the lowest H-bias samples. The H-bias can be calculated by ```python calKDN ``` function in the ```python test_cbt.py ```.


## Fairness Evaluation 
We applied ```python abroca ``` package in [ABROCA](https://pypi.org/project/abroca/). A sample calculation of ABROCA: 
```python
slice = compute_abroca(abrocaDf, 
        pred_col = 'prob_1' , 
        label_col = 'label', 
        protected_attr_col = 'gender',
        majority_protected_attr_val = '2',
        compare_type = 'binary', # binary, overall, etc...
        n_grid = 10000,
        plot_slices = False)
```


<!-- 

|               Function                |                                   Example                                                 |
| :-----------------------------------: | :---------------------------------------------------------------------------------------: |
|            tokenize                   |                ```python tokenize.word_tokenize()    ```                                |
|            stemming                   |                     ```python          defaultdict(lambda : wn.NOUN)    ```            |
|     removing non-alphabetic words     |                    ```python           defaultdict(lambda : wn.NOUN)   ```                |
|           lemmatize                   |                   ```python       WordNetLemmatizer().lemmatize()      ```              |
|      Training/test set split          |                    ```python           CofetEntry.preTrain()           ```        | 


-->



<!-- 


The toolkit is divided into four components: Data adapter, Feature Composer, Model Selector and Performance Evaluator

## Data adapter: 
Data adapter is designed to transform raw input data into a proper format to be used in subsequent steps. The input data is expected to be in a csv file which should include a post text column and a label column. As an example, we have included Stanford forum post dataset used in this study to the toolkit repository. The data adaptor component is responsible for pre-processing the raw text contained in a post (e.g., stemming and removing non-alphabetic words). Then, the pre-processed posts are randomly split into training and testing set according to a pre-defined ratio (i.e., 80% for training and 20% for testing).

e.g., 
```python
cofeter.adapt()
```


|               Function                |                                   Example                                                 |
| :-----------------------------------: | :---------------------------------------------------------------------------------------: |
|            tokenize                   |                ```python tokenize.word_tokenize()    ```                                |
|            stemming                   |                     ```python          defaultdict(lambda : wn.NOUN)    ```            |
|     removing non-alphabetic words     |                    ```python           defaultdict(lambda : wn.NOUN)   ```                |
|           lemmatize                   |                   ```python       WordNetLemmatizer().lemmatize()      ```              |
|      Training/test set split          |                    ```python           CofetEntry.preTrain()           ```        |



## Feature Composer:  
Feature Composer generates a vector representation to represent a post. This vector representation can be used as the input for subsequent classification models. Depending on the selected classification model (traditional ML models vs. DL models), this component either generates a vector consisting of a list of commonly-used features (for traditional ML models), or a embedding-based vector (for DL models). Most of the textual features used in this study (in Section II) are included in the feature composer except for Coh- metrix, LSA similarity and LIWC features as these three features requires external software to generate. However, once generated the features can be easily integrated by simply append the additional feature set to the output feature vector produced by feature composer. As an example,LIWC software9)havetheoptiontoproduce a csv file containing feature set per post, a user may generate LIWC feature using their software and add those features to the output file of this step. For DL models, The embedding vector will be generated using bert-as-a-service, the output will be in 768 dimensional BERT embedding. To enable an efficient evaluation, the generated vectors are stored locally and can be used as input for different models.


|               Function                |                                   Example                                                 |
| :-----------------------------------: | :---------------------------------------------------------------------------------------: |
|            TFIDF                   |                ```python models.util.feature_tfidf    ```                                |
|            NGRAM                   |                     ```python         models.util.feature_ngram    ```            |
|     READABILITY     |                    ```python           models.util.feature_readability ```                |
|           TopkFeatureTEST                   |                   ```python      models.util.exam_selectKbest     ```              |


## Model Selector: 
Model Selector handles the selection of a model. That is, users of this toolkit can choose from any of the four traditional ML models and the five DL models used in this study. We also note that a new model can be easily added and testified under the current framework. After a model is selected, the users can either directly adopt the model parameters derived from our evaluation or training the model from scratch, e.g., using grid search to fine- tune a traditional ML model or coupling BERT with a DL model for co-training.

* initialise base classifier:
```python
classifier = ml_clf(config)
```

* create a Naive bayes classifier: 
```python
classifier.nb_clf()
```

* create a SVM classifier:
```python
classifier.svm_clf()
```

* create a Logistic regression classifier:
```python
classifier.lr_clf()
```

* create a Random forest classifier:
```python
classifier.rf_clf()
```

* to perform a simple grid search with pre-defined parameters:

```python
grid = GridSearchCV(YOUR_MODEL,YOUR_SEARCH_PARAMS,refit=True,verbose=2)
grid.fit(self.Train_X,self.Train_Y)
print(grid.best_estimator_)
```

* to run a model with hyperparamter, replace model function and add parameter: 
e.g., 
replace: 
```python
rfc=RandomForestClassifier()
```

with: 
```python
rfc=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
  max_depth=None, max_features=0.5, max_leaf_nodes=None,
  min_impurity_split=1e-07, min_samples_leaf=1,
  min_samples_split=4, min_weight_fraction_leaf=0.0,
  n_estimators=250, n_jobs=1, oob_score=False, random_state=None,
  verbose=0, warm_start=False)
```

We used a service called [Bert-as-a-service](https://github.com/hanxiao/bert-as-service) to generate BERT embeddings of the forum post. 
The embedding is then used as input for DL models

We refer this repo: [TextClassification](https://github.com/zackhy/TextClassification), where DL code was modified from. 

|               Function                |                                   Example                                                 |
| :-----------------------------------: | :---------------------------------------------------------------------------------------: |
|            Naive bayes                   |                ```python ml_classifiers.nb_clf    ```                                |
|            Logistic regression                   |                     ```python      ml_classifiers.lr_clf    ```            |
|     Random forest     |                    ```python          ml_classifiers.rf_clf ```                |
|           Support vector machine                   |                   ```python    ml_classifiers.svm_clf     ```              |
|           CLSTM                  |                   ```python      clstm_classifier    ```              |
|          BLSTM                   |                   ```python      rnn_classifier     ```              |

## Performance Evaluator: 
Performance Evaluator is responsible for calculat- ing the classification performance of a model in terms of the following four metrics: Accuracy, Cohen’s κ, AUC, and F1 score.


|               Function                |                                   Example                                                 |
| :-----------------------------------: | :---------------------------------------------------------------------------------------: |
|            Accuracy                   |                ```python sklearn.metrics.accuracy_score    ```                                |
|           cohen's Kappa                |                     ```python      sklearn.metrics.cohen_kappa_score    ```            |
|     AUC     |                    ```python          sklearn.metrics.roc_auc_score ```                |
|           F1                   |                   ```python    sklearn.metrics.f1_score     ```              |




 -->
