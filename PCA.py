import numpy as np
import numpy.linalg as la
import pandas as pd
import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy as sp
import seaborn as sns

#get least # comps to accnt 4 90% var
def least_index(pca, percent=.9):
  i=0
  while i<= pca.n_components_:
    if np.sum([pca.explained_variance_ratio_[:i]]) > percent: break
    i+=1
  return i

def err_v_index(pca, p):
  I = least_index(pca, p)
  
  V = pca.components_[:I]
  W = V.dot(A)
  approx = lambda i: W.T[i].dot(V) + MU
  
  err = [approx(i) - A[i] for i in range(I)]
  return err

def integrate(percent_change):
  ps=[0]
  partial_sum=0
  for p in percent_change:
    partial_sump+=p
    ps += [partial_sum]
  return ps

#read data
X = pd.read_csv("File.csv").iloc[:,1:]

#missing nans
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)
T = imp_mean.transform(X)
X = pd.DataFrame(T, columns=X.columns)
#mean
MU = np.mean(X, axis=1)

for ticker in X.columns[:50]:
  ts=X[ticker]
  #T = len(ts)
  plot_acf(ts)
    #plt.plot(range(T+1), integrate(ts+MU))

plt.show()
quit()

#center at zero
A = pd.concat([X[ticker]-MU for ticker in X.columns], axis=1)

pca = PCA()
pca.fit(A.T)

I = least_index(pca, .9)
tot_var_explained = integrate(pca.explained_variance_ratio_)

#eigs
V = pca.components

#approx
W = V.dot(A)

print(W.shape)
W_bar = np.mean(W)
W_std = np.std(W)
print(W_bar, W_std)

for w in W[:50]:
  sns.distplot(w)
plot.show()

variances = [np.std(w) for w in W]
sns.distplot(variances)
plt.show()


for i in range(len(W[0])):
  plt.plot(range(len(W[0])), np.sort(np.abs(W[i])))
plt.title('distribution of weights of eigenstocks')
plt.show()

approx = lambda i: W.T[i].dot(V) + MU

err = np.array([np.abs(approx(i) - A[i]) for i in range(I)])

def plot_var():
  for p in [.8, .9, .95, .98]:
    N = len(tot_var_explained)
    i = 0 
    while i < N:
      if tot_var_explained[i]>p:break
      i+=1
    plt.plot(range(len(tot_var_explained)), tot_var_explained)
    plt.title("total variance explained \nby number of componenets")
    plt.xlabel('number of principal components')
    plt.ylabel('total percentage variation accounted for')
    plt.savefig('var_v_n_components.png')
    plt.show()
    
def show_stocks():
  ticker = 'AAPL'
  i = tickers.tolist().index('AAPL')
  approx = lambda i: W.T[i].dot(V) + MU
  test = approx(i)
  test_int = integrate(test)
  A_int = integrate(A[i])
  plt.plot(range(len(A_int)), A_int, label="original")
  plt.plot(range(len(test_int)), test_int, label="mapping")
  plt.legend()
  plt.title('approximating the stocks {}\nusing first {} components'.format(ticker,I))
  plt.savefig('AAPL_weight.png')
  plt.show()
  
  def plot_approx_err():
    err = err_v_index(pca, 0.9)
    plt.imshow(np.abs(err))
    plt.title('absolute error when \naccounting for 98& variance')
    plt.colorbar()
    plt.savefig('approximation_error_abs.png')
    plt.show()
    
    plt.hist(err, bins=20)
    plt.show()
    err = err_v_index(pca, 0.98)
    plt.imshow(np.abs(err))
    plt.title('absolute error when \naccounting for 98% variance')
    plt.colorbar()
    plt.savefig('approximtion_error_abs1.png')
    plt.show()



















