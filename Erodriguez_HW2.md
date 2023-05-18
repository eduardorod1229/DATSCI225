# ML225 Class - HW2  
  
###  Eduardo Rodriguez  


---  
tags: ['spring2023','machinelearning','datsci225']  
project: HW2  
DiscussedWith:  
date_met: 2023-05-01  
type: Homework  
---  
  
  
**Overview:** In this homework you will implement very popular state of the art algorithms. For each algorithm please evaluate and report performance on the test set for the datasets provided. Use $R^2$ as the metric of comparison. When comparing two algorithms or more please always make predictions with all of them in the same test set and perform a paired-wise t-test comparison of the $R^2$ obtained to find the p value of one algorithm being better than the other.  

### Exercise 2: 40 pts  
  
Please implement two versions of stacking using as base learners: Single individual trees of depth [1,2,3,4,5], linear model. In version A) let the cofficients change freely. In version B) let the coeffcients to take values between 0 and 1.  
Analyze the datasets provided in class. Which version performs best? **HINT:** Change the form of the coeffcients in the linear model to a form that guarantees that their value will always be between 0 and 1.  
  
  
**Import Libraries**  
```python
import time  
import pandas as pd  
import numpy as np  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  

start_time = time.time()  
```

### Data Preparation  
#### Importing Data  

```python
df_names = ["1028_SWD",  
            "1199_BNG_echoMonths",  
            "1201_BNG_breastTumor",  
            "1595_poker",  
            "201_pol",  
            "218_house_8L",  
            "225_puma8NH",  
            "294_satellite_image",  
            "537_houses",  
            "564_fried",  
            "573_cpu_act",  
            "574_house_16H"]
```
  
```python  
import glob  
data_dir = "H:\\Documents\\PhD\\2023_spring\\DATSCI225\\HW2\\pmlb_datasets\\*.tsv.gz"  
file_paths = glob.glob(data_dir)  
datasets = []  
  
for file_path in file_paths:  
      dataset = pd.read_csv(file_path, delimiter='\t')  
      datasets.append(dataset)  
  ```
  
#### Splitting the data in train, validation, and test sets  
```python
seed = 44  
train_datasets = []  
val_datasets = []  
test_datasets= []  
  
for dataset in datasets:  
      #Separating the tests between features and outcomes  
      X = dataset.iloc[:, :-1]  
      y = dataset.iloc[:, -1]  
  
      #Splitting the datasets into train, validation, and test sets  
  
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=seed)  
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=seed)  
  
      #Constructing the datasets  
      train_datasets.append((X_train, y_train))  
      val_datasets.append((X_val, y_val))  
      test_datasets.append((X_test, y_test))  
```

  
  
#### Defining learners for version A  
  
A list of tree regressors plus a linear regressor  
```python
#Initialize the class function that takes a list of base models to train on plus the meta_model  
from sklearn.model_selection import cross_val_predict  
class StackingRegressorA:  
      def __init__(self, base_models=None, meta_model=None):  
            self.base_models = base_models  
            self.meta_model = meta_model  
  
      # fits the training data creating predictions and storing them.  
      def fit(self, X, y):  
            y = np.ravel(y) # this is necessary to change the dimensions on the outcome variable after they are imported from a pd.Datafraem  
            base_predictions = np.zeros((X.shape[0], len(self.base_models)))  
            for i, model in enumerate(self.base_models):  
                  #performing cross-validated new_data_predictions  
                  model.fit(X, y)  
                  base_predictions[:,i] = cross_val_predict(model, X, y, cv=10)  
  
  
            self.meta_model.fit(base_predictions, y)  
            return self  
  
      #Store and return the predictions from the base model to pass it to the meta_learner  
      def predict(self, X):  
            base_predictions = np.zeros((X.shape[0], len(self.base_models)))  
            for i, model in enumerate(self.base_models):  
                  base_predictions[:, i] = model.predict(X)  
  
            #Return the predictions from the meta_learner  
            return self.meta_model.predict(base_predictions)  
  
  
 ``` 
  
### Stacking model B  
*Letting the coefficients to take values between 0 and 1*  

```python
from sklearn.base import BaseEstimator, RegressorMixin  
from sklearn.linear_model import LogisticRegression  
  
class StackingRegressorB(BaseEstimator,RegressorMixin):  
      def __init__(self,base_models=None, meta_model=None):  
            self.base_models = base_models  
            self.meta_model = meta_model  
      # Fitting the base models  
      def fit(self, X, y):  
            y = np.ravel(y) # this is necessary to change the dimensions on the outcome variable after they are imported from a pd.Datafraem  
            base_models_pred = np.zeros((len(X), len(self.base_models)))  
            for i, model in enumerate(self.base_models):  
                  model.fit(X, y)  
                  base_models_pred[:,i] = cross_val_predict(model, X, y, cv=3)  
  
            #Now we fit the meta-model/learner  
            self.meta_model.fit(base_models_pred, y)  
  
            #Forcing the coefficients to be between [0,1]  
  
            self.coef_ = np.clip(self.meta_model.coef_, 0,1)  
  
            return self  
      def predict(self, X):  
  
            #Predict with base tree_models  
            base_models_pred= np.zeros((len(X), len(self.base_models)))  
            for i, model in enumerate(self.base_models):  
                  base_models_pred[:, i] = model.predict(X)  
  
            #Pass those predictions to the meta-model to make predictions  
            meta_model_pred = self.meta_model.predict(base_models_pred)  
  
            return meta_model_pred  
  
  ```
  
#### Wrapping everyting into a function that returns the $R^2$ values  
```python

from sklearn.metrics import r2_score  
  
def fitting_models(X_tr, y_tr, X_val, y_val, stack='A'):  
      # initialize the models with the different depths and train them  
      depths = [1, 2, 3, 4, 5]  
      tree_models = [DecisionTreeRegressor(max_depth=depth, random_state=44) for depth in depths]  
      linear_model = LinearRegression()  
      if stack == 'A':  
            # initialize the stacking models  
  
            stacking_model_A = StackingRegressorA(base_models=[*tree_models, linear_model], meta_model=LinearRegression())  
  
            stacking_model_A.fit(X_tr, y_tr)  
            y_pred = stacking_model_A.predict(X_val)  
            r2_a = r2_score(y_val, y_pred)  
            print("R-squared score: ", r2_a)  
            return r2_a  
      else:  
            # initialize the models with the different depths and train them  
            depths = [1, 2, 3, 4, 5]  
            tree_models = [DecisionTreeRegressor(max_depth=depth, random_state=44) for depth in depths]  
            linear_model = LinearRegression()  
  
            # initialize the stacking models  
            stacking_model_B = StackingRegressorB(base_models=[*tree_models, linear_model], meta_model=LinearRegression())  
            stacking_model_B.fit(X_train, y_train)  
            y_pred = stacking_model_B.predict(X_val)  
            r2_b = r2_score(y_val, y_pred)  
            print("R-squared score: ", r2_b)  
            return r2_b  
  
  ```
  

#### Fitting all the models and getting $R^2$ errors  
```python

r2_stackA = []  
for i in range(len(train_datasets)):  
      X_train, y_train = train_datasets[i]  
      X_val, y_val = val_datasets[i]  
      r2_stackA.append(fitting_models(X_train, y_train, X_val, y_val, stack='A'))  

r2_stackB = []  
for i in range(len(train_datasets)):  
      X_train, y_train = train_datasets[i]  
      X_val, y_val = val_datasets[i]  
      r2_stackB.append(fitting_models(X_train, y_train, X_val, y_val, stack='B'))  

r_2 = pd.DataFrame({'Dataset':df_names})  
r_2["R-squared_A"] = r2_stackA  
r_2["R-squared_B"] = r2_stackB  
r_2["Larger-RS"] = np.where(r_2['R-squared_A']>r_2['R-squared_B'], ' A','B')  
r_2  

#### Paired t-test  
```python
from scipy.stats import ttest_rel  
def model_compare(model1, model2):  
      t_test, p_value = ttest_rel(r_2[model1], r_2[model2])  
      diff = r_2[model1] - r_2[model2]  
      mean_diff = diff.mean()  
      print('T-statistic:', t_test)  
      print('P-value: ',p_value)  
      print("Mean difference: ",mean_diff)  
      
  
model_compare('R-squared_A', 'R-squared_B')
```


  
## 3 Excerise 3: 10 pts  
Please implement Random Forests using a combination of decision trees and linear models. On each iteration 1) randomly choose the observations using bootstrap 2) randomly choose whether you are going to use a tree or a linear model 3) randomly choose the number of features included. You can use any package of your choice to build the individual trees and linear models. Random Forests randomly samples the features used on every split of the decision tree but to facilitate implementation use the same features for all the partitions of a given tree. Analyze the datasets provided it and compare performance to stacking.  

```python

import random  
from sklearn.utils import resample  
  
class RandomForestRegressor(BaseEstimator):  
      def __init__(self, n_trees, max_depth=None, n_features=None):  
            self.n_trees = n_trees  
            self.max_depth = max_depth  
            self.n_features = n_features  
            self.trees = []  
            self.linear_models = []  
  
      def fit(self, X, y):  
            n_samples, n_features = X.shape  
            if self.max_depth is None:  
                  self.max_depth = int(np.log2(n_features))  
            if self.n_features is None:  
                  self.n_features = random.randint(1, n_features)  
  
            y = np.ravel(y)  
  
            for _ in range(self.n_trees):  
  
                  X_boot, y_boot = resample(X, y, replace=True)  
  
                  # Randomly choose features using random feature sampling  
                  selected_features = random.sample(range(n_features), self.n_features)  
                  X_boot_selected = X_boot[:, selected_features]  
  
                  if random.choice([True, False]):  
  
                        tree = DecisionTreeRegressor(max_depth=self.max_depth)  
                        tree.fit(X_boot_selected, y_boot)  
                        self.trees.append(tree)  
                        self.linear_models.append(None)  
                  else:  
                        linear_model = LinearRegression()  
                        linear_model.fit(X_boot_selected, y_boot)  
                        self.trees.append(None)  
                        self.linear_models.append(linear_model)  
  
      def predict(self, X):  
            predictions = []  
            for i in range(self.n_trees):  
                  if self.trees[i] is not None:  
  
                        selected_features = self.trees[i].tree_.feature[:self.n_features]  
                        X_selected = X[:, selected_features]  
                        predictions.append(self.trees[i].predict(X_selected))  
                  else:  
  
                        selected_features = range(self.n_features)  
                        X_selected = X[:, selected_features]  
                        predictions.append(self.linear_models[i].predict(X_selected))  
            return np.mean(predictions, axis=0)  
  ```
  
**Evaluating the model**  

```pyhon

  
def forest_eval(i):  
      X_train, y_train = train_datasets[i]  
      X_val, y_val = val_datasets[i]  
  
      X_train = np.array(X_train)  
      y_train = np.array(y_train)  
      X_val = np.array(X_val)  
      y_val = np.array(y_val)  
  
      #Initialize the model  
      random_forest = RandomForestRegressor(n_trees=10, max_depth=None, n_features=None)  
  
     #Fit the model  
      random_forest.fit(X_train, y_train)  
  
      #Make predictions  
      y_pred = random_forest.predict(X_val)  
      #Obtain R-squared  
      r2_rf = r2_score(y_val, y_pred)  
      print("R-squared score: ", r2_rf)  
  
      #Returned the y_pred variable to check because it was doing funky stuff  
      return r2_rf, y_pred  
```

  
#### Obtaining $R^2$  
```python

r2_rfs = []  
for i in range(len(train_datasets)):  
      r2_rf , y_pred= forest_eval(i)  
      r2_rfs.append(r2_rf)  
```

  
#### Appending it to the metrics DF for evaluation  
```python

r_2['R-squared_RF'] = r2_rfs  
r_2  
```

  
#### Paired t-test  
```python
  
#Compared to Stack A  
  
model_compare('R-squared_A', 'R-squared_RF')  
```
```python
#Compared to Stack B  
  
model_compare('R-squared_B', 'R-squared_RF')  
```
```python
#Just for reference  
model_compare('R-squared_A', 'R-squared_B')  
```

Overall, extremely negligible difference between the difference in Stack B and A compared to RF. Stack is performing better (marginally)  
  
## 4 Excerise 4: 10  
Implement Gradient Boosting for regression using number of trees, tree depth and learning rate as hyper-parameters. Analyze the datasets provided above and compare performance using the three algorithms (Stacking, Gradient Boosting, Random Forests).  
  
```python  
class GradientBoostingRegressor:  
      def __init__(self, n_trees, max_depth, learning_rate):  
            self.n_trees = n_trees  
            self.max_depth = max_depth  
            self.learning_rate = learning_rate  
            self.trees = []  
            self.intercept = None  
  
      def fit(self, X, y):  
            # Initialize prediction to the mean of y  
            self.intercept = np.mean(y)  
            y_pred = np.full_like(y, self.intercept)  
            for i in range(self.n_trees):  
                  # Calculate residuals  
                  residuals = y - y_pred  
                  # Train decision tree on residuals  
                  tree = DecisionTreeRegressor(max_depth=self.max_depth)  
                  tree.fit(X, residuals)  
                  # Update predictions with learning rate and tree prediction  
                  y_pred += self.learning_rate * tree.predict(X)  
                  # Store tree in list of trees  
                  self.trees.append(tree)  
  
      def predict(self, X):  
            # Make predictions by summing predictions from all trees and adding the intercept  
            predictions = np.full(X.shape[0], self.intercept)  
            for tree in self.trees:  
                  predictions += self.learning_rate * tree.predict(X)  
            return predictions  
  

def gb_eval(i):  
      X_train, y_train = train_datasets[i]  
      X_val, y_val = val_datasets[i]  
  
      X_train = np.array(X_train)  
      y_train = np.array(y_train)  
      X_val = np.array(X_val)  
      y_val = np.array(y_val)  
  
      #Initialize the model  
      gb_model = GradientBoostingRegressor(n_trees=100, max_depth=5, learning_rate=0.1)  
  
      #Fit the model  
      gb_model.fit(X_train, y_train)  
  
      #Make predictions  
      y_pred = gb_model.predict(X_val)  
      #Obtain R-squared  
      r2_gb = r2_score(y_val, y_pred)  
      print("R-squared score: ", r2_gb)  
  
      #Returned the y_pred variable to check because it was doing funky stuff  
      return r2_gb, y_pred  



r2_gbs = []  
for i in range(len(train_datasets)):  
      r2_gb , y_pred= gb_eval(i)  
      r2_gbs.append(r2_gb)  
r_2['R-squared_GB'] = r2_gbs  
r_2  
  
model_compare('R-squared_A','R-squared_RF')  
  
model_compare('R-squared_B','R-squared_RF')  
  
model_compare('R-squared_RF','R-squared_GB')  
  
model_compare('R-squared_A','R-squared_GB')  
  
model_compare('R-squared_B','R-squared_GB')  
  
end_time = time.time()  
execution_time = end_time - start_time  
print('Execution time: ', execution_time/60, "mins")  

```
