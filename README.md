# Kaggle Competition

## Data
The data comes from micro-RNA expression levels (columns `['v0': 'v977']`) by subject. 
`target` variable is a categorical variable that marks the outcome. *Note: even though for all subjects there is at least a target 0 and 1. they do not mean the same thing (see `problem_id` below)*

`probelm_id` variable marks the task to be performed. Each `target` variable has a different meaning depending on `probelm_id`. We do not have access to the meaning of each level. 

## Data Preparation

### Import libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



```
### Import Data
```python

df = pd.read_csv('train.zip', index_col= None)

#gene_list = ['v' + str(i) for i in range(978)] (optional) This variable is a place holder that can be used to parse through the columns
```

Given that the Big dataset is comprised of multiple subsets, is better to put it on a list or a dictionary so the models can be fitted to each dataset (task). We can do a quick normalization on the data, after a few iterations 'shallow' normalization helped performance


```python
subset_dataframes = {}

for problem_id in df['problem_id'].unique():
    subset_df = df[df['problem_id'] == problem_id].copy()

    # Select the columns to be normalized
    columns_to_normalize = subset_df.loc[:, 'v0':'v977'].columns

    # Apply normalization using StandardScaler
    scaler = StandardScaler()
    subset_df.loc[:, columns_to_normalize] = scaler.fit_transform(subset_df.loc[:, columns_to_normalize])

    subset_dataframes[f'df_{problem_id}'] = subset_df
```

Do the same for test data

```python

df_test = pd.read_csv('test0.csv')
subset_df_test = {}
for problem_id in df_test['problem_id'].unique():
    subset_df = df_test[df_test['problem_id'] == problem_id].copy()
    columns_to_normalize = subset_df.loc[:, 'v0':'v977'].columns
    scaler = StandardScaler()
    subset_df.loc[:, columns_to_normalize] = scaler.fit_transform(subset_df.loc[:, columns_to_normalize])
    subset_df_test[f'df_{problem_id}'] = subset_df
```

Now the datasets can be access selecting the `df_` + `problem_id`. For example, to access the first dataframe: `subset_dataframes['df_0']` or `df_test['df_0']` for training and testing data, respectively.

## Basic Linear model (benchmarking)

Let's do a simple "vanilla" linear model to see where are we standing: 

```python

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#Create an empty dictionary where the parameters of linear models will be stored.
linear_models = {}

# This loops over the dataframes in the dictionary that we created before

for key, dataset in subset_dataframes.items():
  X= dataset.loc[:,'problem_id':'v977']
  y= dataset['target']
  
  # The argument 'stratify=y' tries to keep the same proportion of labels in the training and the testing data (because we have imbalanced data)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
  
  #Initialize the mode
  linear_model = LinearRegression()
  
  #Fit the model
  linear_model.fit(X_train, y_train)
  
  # Make predictions
  y_pred = linear_model.predict(X_test)

  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  
  #Append the dictionary with the parameters of the linear model and the model itself.
  linear_models[key] = {'model': linear_model, 'params': linear_model.coef_}
  
```

After we fit our model we want to check with the "test" data from the comptetition


```python

# Initialize an empty dictionary to keep the predictions
lin_preds= {}

#Iterate through the datasets
for key, dataframe in subset_df_test.items():
    
    #Select the 'features'
    X_new = dataframe.loc[:, 'problem_id':'v977']
    
    # Initialize the model that was best for each dataframe
    model = linear_models[key]['model']
    
    # Make predictions
    y_pred = model.predict(X_new)
    
    # Store the predictions in the dictionary
    lin_preds[key] = y_pred

```

Put the predictions on a single dataframe suitable to upload to Kaggle


```python
predictions_df = pd.DataFrame()
for key, predictions in lin_preds.items():
    obs_ids = subset_df_test[key]['obs_id']  
    
    dataset_predictions_df = pd.DataFrame({'obs_id': obs_ids, 'target': predictions})
    
    predictions_df = pd.concat([predictions_df, dataset_predictions_df], ignore_index=True)
```

After that benchmark we can repeat itteratively with other models that we believe are suitable to test. 

Here's the final model that I used


```python
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

stacked_models = {}

def train_model(key, dataset):
    X = dataset.loc[:, 'problem_id':'v977']
    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84, stratify=y)


#Random forest and elastic ned will capture non-linearities
    random_forest_model = ExtraTreesClassifier() #RandomForestClassifier(n_estimators=150, random_state=84, max_features='auto', min_samples_leaf=5, max_depth = 20)
    elastic_net_model = ElasticNet(random_state=82, selection='random', max_iter=10000, fit_intercept=False, warm_start=True, alpha = 0.01, l1_ratio=0.1)

    #One logistic regression model to capture linearities and put it in the context of classification problems
    logistic_regression_model1 = LogisticRegression(random_state=82)
    
    #After testing a bunch of models (including trees) the best final estimator is a LogisticRegressor
    logistic_regression_model2 = LogisticRegression(random_state=82)
    
    
    # Perform Grid search for the logistic models
     param_grid = {
        'C': [0.1, 1.0, 10.0],  
        'penalty': ['l1', 'l2'] 
    }

    # CV 8 is the best balance point between fast fitting and higher accuracy
    grid_search1 = GridSearchCV(logistic_regression_model1, param_grid, scoring='accuracy', cv=8)
    grid_search1.fit(X_train, y_train)

    best_logistic_regression_model1 = grid_search1.best_estimator_
    grid_search2 = GridSearchCV(logistic_regression_model2, param_grid, scoring='accuracy', cv=8)
    grid_search2.fit(X_train, y_train)

    best_logistic_regression_model2 = grid_search2.best_estimator_


    # Build the final stack
    stacked_model = StackingClassifier(
        estimators=[
                    ('random_forest', random_forest_model),
                    ('elastic_net', elastic_net_model),
                    ('logistic_regression1', best_logistic_regression_model1),
                    ],
        final_estimator=best_logistic_regression_model2
    )
    stacked_model.fit(X_train, y_train)
    y_pred = stacked_model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None).round().astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    return key, {'model': stacked_model, 'accuracy': accuracy, 'predictions': y_pred}


#This library is to run everything in parallel so it takes less time
results = Parallel(n_jobs=-1)(delayed(train_model)(key, dataset) for key, dataset in subset_dataframes.items())

stacked_models = dict(results)

```
Check the models accuracies and take the mean of all the predictions ( a rough estimate)

```python

accuracies = [model_info['accuracy'] for model_info in stacked_models.values()]
print('Individual accuracies \n', accuracies)
print("Overall mean accuracy: ",np.mean(accuracies))
```


Now use the models to make predictions in our testing data:

```python

stacking_preds= {}
for key, dataframe in subset_df_test.items():
    X_new = dataframe.loc[:, 'problem_id':'v977']
    
    model = stacked_models[key]['model']
    
    y_pred = model.predict(X_new)
    
    stacking_preds[key] = y_pred

# Create a dataframe for the predictions
predictions_df = pd.DataFrame()
for key, predictions in stacking_preds.items():
    obs_ids = subset_df_test[key]['obs_id']  
    
    dataset_predictions_df = pd.DataFrame({'obs_id': obs_ids, 'target': predictions})
    
    predictions_df = pd.concat([predictions_df, dataset_predictions_df], ignore_index=True)

    
predictions_df = predictions_df.sort_values('obs_id')
predictions_df['target'] = predictions_df['target'].astype(int)

#save the dataframe
predictions_df.to_csv('stacking_predictions2.csv',index=False)
```


## Multioutput prediction

It is possible to write 'easier' models to perform multioutput modelling or check Multi-tasking learning more detail separately. 

For this type of modeling, we need to encode the labels into multiple binary columns. This is, we will create a column for each value in the `target` variable from [0:4]. So datasets with w values (0 and 1) will have two columns and datasets with values [0:4] will have 5 columns 

```python

rom sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed


#Encode the targets 
unique_levels = df['target'].unique()
label_encoder = LabelEncoder()

mo_preds = {}

def train_model(problem_id, subset_df):
    X = subset_df.loc[:, 'problem_id':'v977']  
    y = pd.DataFrame(np.nan, index=subset_df.index, columns=unique_levels)

    for level in unique_levels:
        binary_target = label_encoder.fit_transform(subset_df['target'] == level)
        y[level] = binary_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84, stratify=y)

    base_classifier = LogisticRegression(random_state=84)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [2,5, 10],
        'max_depth': [None, 10, 20]
    }

    grid_search = GridSearchCV(base_classifier, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    best_base_classifier = grid_search.best_estimator_

    multi_output_classifier = MultiOutputClassifier(best_base_classifier)

    multi_output_classifier.fit(X_train, y_train)

    y_pred = multi_output_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return problem_id, {
        'model': multi_output_classifier,
        'accuracy': accuracy,
        'predictions': y_pred
    }

results = Parallel(n_jobs=-1)(delayed(train_model)(problem_id, subset_df) for problem_id, subset_df in subset_dataframes.items())

mo_preds = dict(results)


```

## Next steps

Fit all the models on training data for hyper-parameter tunning and once it is found, fit the model on the full dataset.
