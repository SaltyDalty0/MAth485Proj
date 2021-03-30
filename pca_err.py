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


# gets least number of components_ to account for 90% of the variance
def least_index(pca, percent=.9):
    i = 0
    while i <= pca.n_components_:
        if np.sum([pca.explained_variance_ratio_[:i]]) > percent:  break
        i += 1
    return i

def err_v_index(pca, p):
    I = least_index(pca, p)

    V = pca.components_[:I]
    W = V.dot(A)
    approx = lambda i: W.T[i].dot(V) + MU

    err = [approx(i) - A[i] for i in range(I)]
    return err

def integrate(percent_change):
    ps = [0]
    partial_sum = 0
    for p in percent_change:
        partial_sum+=p
        ps.append(partial_sum)
    return ps

#file = "stock_data_17-21.csv"
#data = pd.read_csv(file)

#tickers = np.unique(data['Name'].values)

# read raw data
X = pd.read_csv("C:/Users/LENOVO/Desktop/PCA_485/sp500_joined_closes.csv", index_col=0)#.iloc[:,1:]
X = X.pct_change()

# impute missing nan data: replace nan using SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)
T = imp_mean.transform(X)
X = pd.DataFrame(T, columns=X.columns)
# compute the mean
MU = np.mean(X, axis=1)

for ticker in X.columns[:50]:
    ts = X[ticker]
    T = len(ts)
    plot_acf(ts)
#     plt.plot(range(T+1), integrate(ts+MU))
#
plt.show()
quit()


# recenter the data
A = pd.concat([X[ticker]-MU for ticker in X.columns], axis=1)

pca = PCA()
pca.fit(A.T)

I = least_index(pca, .9)
tot_var_explained = integrate(pca.explained_variance_ratio_)

# The eigenstocks
V = pca.components_

# The approximation
W = V.dot(A)

print(W.shape)
W_bar = np.mean(W)
W_std = np.std(W)
print(W_bar, W_std)
# plt.plot(range(pca.n_components_),pca.explained_variance_ratio_)
# plt.xlabel('number of components')
# plt.ylabel('explained percent variance')
# plt.savefig('percent_variance_per_eigenstock.png')
# # for i in range(100):
# #     plt.plot(range(len(W[0])), np.sort(W[i]),'.')
# # plt.hist(V[0], bins=100)
#
# plt.show()
for w in W[:50]:
    sns.distplot(w)
plt.show()

variances = [np.std(w) for w in W]
sns.distplot(variances)
plt.show()
# logW = pd.DataFrame(np.log(np.abs(W)))
# tau = logW[logW>.001]
# plt.imshow(tau)
# plt.colorbar()

for i in range(len(W[0])):
    plt.plot(range(len(W[0])), np.sort(np.abs(W[i])))
    # plt.plot(range(len(W[0])), W[:,i])
plt.title('distribution of weights of eigenstocks')
# plt.show()

approx = lambda i: W.T[i].dot(V) + MU

err = np.array([np.abs(approx(i) - A[i]) for i in range(I)])

# plt.plot(range(len(err)), np.log(err))
# plt.show()


def plot_var_v_num_components():
    for p in [.8, .9, .95, .98]:
        N = len(tot_var_explained)
        i = 0
        while i < N:
            if tot_var_explained[i]>p: break
            i+=1
        plt.plot(i, tot_var_explained[i], '*b')
        plt.annotate('N = {}\nV = {:.3f}'.format(i, tot_var_explained[i]),(i, tot_var_explained[i]-.1))
    plt.plot(range(len(tot_var_explained)), tot_var_explained)
    plt.title('total variance explained \nby number of components')
    plt.xlabel('number of principal components')
    plt.ylabel('total percentage variation accounted for')
    plt.savefig('var_v_n_components.png')
    plt.show()

def recreating_stocks():
    ticker = 'AAPL'
    i = tickers.tolist().index('AAPL')
    approx = lambda i: W.T[i].dot(V) + MU
    test = approx(i)
    test_int = integrate(test)
    A_int = integrate(A[i])
    plt.plot(range(len(A_int)), A_int, label= "original")
    plt.plot(range(len(test_int)), test_int, label="recreation")
    plt.legend()
    plt.title('approximating the stock {}\nusing first {} components'.format(ticker,I))
    plt.savefig('AAPL_recreation.png')
    plt.show()

def plot_approx_err():
    err = err_v_index(pca, 0.9)
    plt.imshow(np.abs(err))
    plt.title('absolute error when \naccounting for 90% of variance')
    plt.colorbar()
    plt.savefig('approximation_error_abs.png')
    plt.show()

    plt.hist(err, bins=20)
    plt.show()
    err = err_v_index(pca, 0.98)
    plt.imshow(np.abs(err))
    plt.title('absolute error when \naccounting for 98% of variance')
    plt.colorbar()
    plt.savefig('approximation_error_abs1.png')
    plt.show()


def construct_data_matrix():
    # read data
    file = "stock_data_17-21.csv"
    data = pd.read_csv(file)

    tickers = np.unique(data['Name'].values)

    dfs = []
    for ticker in tickers:
        print(ticker)
        df = pd.DataFrame(data[data['Name']==ticker])
        # print(df['Adj Close'])

        percent_change = lambda df: 100*(df['Close']-df['Open'])/df['Open']
        processed = percent_change(df.reset_index())
        dfs.append(processed)

    df = pd.concat(dfs, axis=1)
    df.columns = tickers
    df.to_csv("percent_change_data_matrix.csv")


def construct_data_matrix_from_old_data():
    data_dir = '/Users/rubyabrams/Dropbox/stochastic_optimization/data/'
    sys.path.append(data_dir)

    X = []
    library = {}
    for stock in os.listdir(data_dir):
        data = pd.read_csv(data_dir+stock)
        percent_change = 100*(data['Close']-data['Open'])/data['Open']
        X.append(percent_change)
        library[stock.replace('.csv','')] = percent_change.values

    X = pd.concat(X, axis=1)

def plot_first_10_eigen_stocks(pca):
    for i in range(10):
        value = integrate(pca.components_[i])
        plt.plot(range(len(value)), value, label="{}^th PC".format(i+1))
    plt.title('First 10 eigenstocks')
    plt.legend(loc='upper left')
    plt.savefig('first_10_eigenstocks.png')
    plt.show()
