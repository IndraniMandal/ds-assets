'''
dsutils - Utility modules for the "Programming for Data Science" course
          at the University of Rhode Island.
'''

import os
import subprocess
if 'google.colab' in os.sys.modules:
  subprocess.run(['pip3','install','PyMySQL'])

import warnings
import math
import pandas as pd
import statistics as stats
import numpy # percentile
from sklearn import cluster # KMeans
from sklearn.base import clone
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot # show, xlabel, ylabel
from scipy.spatial import distance # cdist
import pymysql as sql



def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    if acc > 1.0 or acc < 0.0:
      raise ValueError('Expected an accuracy value between 0 and 1')
    # if acc == 1.0 pertub the acc for the lower
    # bound so we get something reasonable in terms of interval
    if acc == 1.0:
      lbacc = 0.99
    else:
      lbacc = acc
    lb = max(0, acc - 1.96*math.sqrt(lbacc*(1-lbacc)/n))
    ub = min(1.0, acc + 1.96*math.sqrt(acc*(1-acc)/n))
    return (lb,ub)

def regression_confint(rs_score, n, k):
    '''
    Compute the 95% confidence interval for a regression problem.
      rs_score -- R^2 score
      n        -- number of observations used to compute the R^2 score
      k        -- number of independent variables in dataset
    Returns a tuple (lb,ub)

    Reference:
    https://books.google.com/books?id=gkalyqTMXNEC&pg=PA88#v=onepage&q&f=false
    '''
    interval = 2*math.sqrt((4*rs_score*(1-rs_score)**2*(n-k-1)**2)/((n**2 - 1)*(n+3)))
    lb = max(0, rs_score - interval)
    ub = min(1.0, rs_score + interval)
    return (lb,ub)

def plot_elbow(X, n=10):
   '''
   KMeans Elbow plot:
     X - feature matrix
     n - max number of centroids to consider

   inspired by the source code from
   https://www.slideshare.net/SarahGuido/kmeans-clustering-with-scikitlearn
   '''
   # flush the display stack
   pyplot.show()
    
   # kmeans models for each k
   kMeansModels = [cluster.KMeans(n_clusters=k, n_init='auto').fit(X) for k in range(1, n+1)]
    
   # coordinates of the centroids of the models
   centroids = [m.cluster_centers_ for m in kMeansModels]
   
   # find the distances of the values to the centroids
   k_euclid = [distance.cdist(X, cent) for cent in centroids]
   
   # find the distance of each point to its cluster center
   dist = [numpy.min(ke, axis=1) for ke in k_euclid]
   
   # average variance for each cluster configuration
   dist_tuple = zip(list(range(1,n+1)),dist)
   average_var = [sum(d**2)/k for (k,d) in dist_tuple]

   # plot the variance of the models
   sns.lineplot(x=list(range(1,n+1)), y=average_var)
   pyplot.xlabel('k')
   pyplot.ylabel('Average Cluster Variance')
   pyplot.show()


def bootstrap(model, X, y):
    '''
    Compute a bootstrapped model score together with its 95% probability 
    bound. If the model object is a classification model then model accuracy 
    is computed and if the model object is a regression model then the R^2 
    score is computed.

    Parameters
        model - a classification/regression model
        X - sklearn style feature matrix
        y - sklearn style target vector
    
    Returns
        (score, lower bound, upper bound) 
    '''
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    D = pd.concat([X,y], axis=1)
    score_list = []
    for i in range(200):
        B = D.sample(n=X.shape[0],
                     axis=0,
                     replace=True,
                     random_state=i)
        BX = B.drop(columns=y.columns)
        By = B[y.columns]
        bootmodel = clone(model).fit(BX, By)
        acc = bootmodel.score(BX,By)
        score_list.append(acc)
    score_avg = stats.mean(score_list)
    score_list.sort()
    score_ub = numpy.percentile(score_list,97.5)
    score_lb = numpy.percentile(score_list,2.5)
    return (score_avg, score_lb, score_ub)


class DBCredentials:
  '''
  Allows the user to instantiate DB credential objects for the
  use in the 'execute_query' function.

  The defaults are set to point to the default DB for the CSC310
  course at URI.
  '''
  def __init__(self,
               host = 'testdb.cwy05wfzuxbv.us-east-1.rds.amazonaws.com',
               userdb = 'world',
               user = 'csc310',
               password = 'csc310$is$fun'):
    self.host = host
    self.userdb = userdb
    self.user = user
    self.password = password

def execute_query(credentials, sql_string):
  '''
  execute_query
    credentials - a instantiated DBCredentials object
    sql_string  - a string holding the SQL query
  
  Returns a Pandas dataframe holding the result table
  '''
  db = sql.connect(host=credentials.host,
                   user=credentials.user,
                   password=credentials.password,
                   database=credentials.userdb)
  warnings.filterwarnings('ignore')
  data = pd.read_sql(sql_string, con=db)
  warnings.filterwarnings('always')
  db.close()
  return data

if __name__ == '__main__':
  from sklearn import tree

  # bootstrap: classification
  t1c = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
  df = pd.read_csv("abalone.csv")
  X = df.drop(columns=['sex'])
  y = df[['sex']]
  print("Confidence interval max_depth=3: {}".format(bootstrap(t1c,X,y)))

  # elbow plot
  df = pd.read_csv("iris.csv")
  X = df.drop(columns=['Species'])
  plot_elbow(X)

