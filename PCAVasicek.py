import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import quandl

with open(r'key.txt', 'r') as fp:
    api_key = fp.readline()
quandl.ApiConfig.api_key = api_key
swap_names = ['FRED/DSWP1', 
             'FRED/DSWP2', 
             'FRED/DSWP3', 
             'FRED/DSWP4', 
             'FRED/DSWP5', 
             'FRED/DSWP7', 
             'FRED/DSWP10', 
             'FRED/DSWP30']

swap_df = quandl.get(swap_names)
swap_df = swap_df.dropna()
swap_df.columns = ["SWAP1",
                  "SWAP2",
                  "SWAP3",
                  "SWAP4",
                  "SWAP5",
                  "SWAP7",
                  "SWAP10",
                  "SWAP30"]
print(swap_df.head())
swap_df2 = swap_df.copy()
swap_df.plot(figsize=(10,5))
plt.ylabel("Rate")
plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2)
plt.title("Original Dataset")
plt.show()
sns.heatmap(swap_df.corr())
plt.show()
df1_returns = swap_df.pct_change()
df1_returns.plot(figsize=(15,8))
plt.title('USA SWAP rates daily absolute returns')
plt.show()
def PCA(df, num_reconstruct):
    df -= df.mean(axis=0)
    R = np.cov(df, rowvar=False)
    eigenvals, eigenvecs = sp.linalg.eigh(R)
    eigenvecs = eigenvecs[:, np.argsort(eigenvals)[::-1]]
    eigenvals = eigenvals[np.argsort(eigenvals)[::-1]]
    eigenvecs = eigenvecs[:, :num_reconstruct]

    return np.dot(eigenvecs.T, df.T).T, eigenvals, eigenvecs
scores, evals, evecs = PCA(swap_df, 7)
evalsum=sum(evals)
evalPercent=[]
for value in evals:
    # Add the value to the new list
    evalPercent.append(value/evalsum)
plt.bar(range(len(evalPercent)), evalPercent)
for i, v in enumerate(evalPercent):
    plt.text(i, v + 0.01, "{:.3%}".format(v), ha='center')
# Add labels and title
plt.xlabel('Principal Component')
plt.ylabel('Explanation')
plt.title('Contribution of Principal Components (swap rates)')
plt.show()
evecs = pd.DataFrame(evecs)
for i, col in enumerate(evecs.loc[:, 0:2]):
    plt.plot(evecs[col], label=f'PC {i+1}')
plt.legend()
plt.legend(loc='upper middle')
plt.title('Top 3 PCs of Swap Rates')
plt.show()
for i, col in enumerate(evecs.loc[:, 3:5]):
    plt.plot(evecs[col], label=f'PC {i+4}')
plt.legend()
plt.legend(loc='upper middle')
plt.title('The 4th to 6th PCs of Swap Rates')
plt.show()
reconst = pd.DataFrame(np.dot(scores,evecs.T), index=swap_df.index, columns=swap_df.columns)

plt.plot(reconst)
plt.ylabel("Rate")
plt.title("Reconstructed Mean-Subtracted Dataset from PCs")
plt.show()
for cols in reconst.columns:
    reconst[cols] = reconst[cols] + swap_df2.mean(axis=0)[cols]
    
reconst.plot(figsize=(10,5))
plt.ylabel("Rate")
plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2)
plt.title("Reconstructed Dataset from PCs")
plt.show()

plt.plot(evals)
plt.ylabel("Contribution to Variance")
plt.xlabel("Principal Component")
plt.show()

import sklearn.decomposition.pca as PCA
treasury = ['FRED/DGS1MO',
           'FRED/DGS3MO',
           'FRED/DGS6MO',
           'FRED/DGS1',
           'FRED/DGS2',
           'FRED/DGS3',
           'FRED/DGS5',
           'FRED/DGS7',
           'FRED/DGS10',
           'FRED/DGS20',
           'FRED/DGS30']

treasury_df = quandl.get(treasury)
treasury_df.columns = ['TRESY1mo',
                    'TRESY3mo', 
                    'TRESY6mo',
                    'TRESY1y',
                    'TRESY2y',
                    'TRESY3y',
                    'TRESY5y',
                    'TRESY7y',
                    'TRESY10y',
                    'TRESY20y',
                    'TRESY30y']
treasury_df.plot(figsize=(10,5))
plt.ylabel("Rate")
plt.legend(bbox_to_anchor=(1.01,.9), loc=2)
plt.show()
sns.heatmap(treasury_df.corr())
plt.show()

treasury_df = treasury_df.astype(float)
treasury_df2 = treasury_df.iloc[:, 3:-2]
treasury_df2 = treasury_df2.dropna()
comb_df = treasury_df2.merge(swap_df2, left_index=True, right_index=True)
pca_t = PCA.PCA(n_components=6)
pca_t.fit(treasury_df2)
PCA.PCA(copy=True, iterated_power='auto', n_components=6, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
plt.plot(pca_t.explained_variance_ratio_)
plt.ylabel("Explained Variance")
plt.xlabel("Principal Component")
plt.show()

plt.plot(pca_t.components_[0:3].T)
plt.title('Top 3 PCs of Treasury Rates')
plt.xlabel("Principal Component")
plt.show()

spread = [comb_df.SWAP1-comb_df.TRESY1y, 
        comb_df.SWAP2-comb_df.TRESY2y,
       comb_df.SWAP3-comb_df.TRESY3y,
       comb_df.SWAP5-comb_df.TRESY5y,
       comb_df.SWAP7-comb_df.TRESY7y,
       comb_df.SWAP10-comb_df.TRESY10y]
spread_df = pd.DataFrame(np.array(spread).T, index=comb_df.index, 
                         columns = ["SPREAD1y", "SPREAD2y", "SPREAD3y", "SPREAD5y", "SPREAD7y", "SPREAD10y"])
spread_df.plot()
plt.ylabel("Swap Spread Over Treasury")
plt.show()
def VasicekNextRate(r, kappa, theta, sigma, dt=1/252):
    # Implements above closed form solution     
    val1 = np.exp(-1*kappa*dt)
    val2 = (sigma**2)*(1-val1**2) / (2*kappa)
    out = r*val1 + theta*(1-val1) + (np.sqrt(val2))*np.random.normal()
    return out
def VasicekSim(N, r0, kappa, theta, sigma, dt = 1/252):
    short_r = [0]*N # Create array to store rates     short_r[0] = r0 # Initialise rates at $r_0$     
    for i in range(1,N):
        short_r[i] = VasicekNextRate(short_r[i-1], kappa, theta, sigma, dt)
    return short_r
def VasicekMultiSim(M, N, r0, kappa, theta, sigma, dt = 1/252):
    sim_arr = np.ndarray((N, M))
    for i in range(0,M):
        sim_arr[:, i] = VasicekSim(N, r0, kappa, theta, sigma, dt)
    return sim_arr
def VasicekCalibration(rates, dt=1/252):
    n = len(rates)
    # Implement MLE to calibrate parameters     
    Sx = sum(rates[0:(n-1)])
    Sy = sum(rates[1:n])
    Sxx = np.dot(rates[0:(n-1)], rates[0:(n-1)])
    Sxy = np.dot(rates[0:(n-1)], rates[1:n])
    Syy = np.dot(rates[1:n], rates[1:n])
    theta = (Sy * Sxx - Sx * Sxy) / (n * (Sxx - Sxy) - (Sx**2 - Sx*Sy))
    kappa = -np.log((Sxy - theta * Sx - theta * Sy + n * theta**2) / (Sxx - 2*theta*Sx + n*theta**2)) / dt
    a = np.exp(-kappa * dt)
    sigmah2 = (Syy - 2*a*Sxy + a**2 * Sxx - 2*theta*(1-a)*(Sy - a*Sx) + n*theta**2 * (1-a)**2) / n
    sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
    r0 = rates[n-1]
    return [kappa, theta, sigma, r0]
params = VasicekCalibration(spread_df.loc[:, 'SPREAD10y'].dropna()/100)
kappa = params[0]
theta = params[1]
sigma = params[2]
r0 = params[3]
print(f"\n\nCalibrated parameters: kappa={kappa:.3f}  theta={theta:.3f}  sigma={sigma:.3f}  r0={r0:.3f}")
years = 1
N = years * 252
t = np.arange(0,N)/252

test_sim = VasicekSim(N, r0, kappa, theta, sigma, 1/252)
plt.plot(t,test_sim)
plt.title("One path for the simulation")
plt.show()
M = 100
rates_arr = VasicekMultiSim(M, N, r0, kappa, theta, sigma)

plt.plot(t,rates_arr)
plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
plt.annotate('Theta', xy=(1.0, theta+0.0005))
plt.xlim(-0.05, 1.05)
plt.ylim([-0.01, 0.01])
plt.ylabel("Rate")
plt.xlabel("Time (yr)")
plt.title("Simulation with calibrated parameters")
plt.show()

M = 100
rates_arr = VasicekMultiSim(M, N, r0, kappa*5, theta, sigma)
plt.plot(t,rates_arr)
plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
plt.annotate('Theta', xy=(1.0, theta+0.0005))
plt.xlim(-0.05, 1.05)
plt.ylim([-0.01, 0.01])
plt.ylabel("Rate")
plt.xlabel("Time (yr)")
plt.title("Kappa scaled up 5 times")
plt.show()
M = 100
rates_arr = VasicekMultiSim(M, N, r0, kappa, theta, sigma*5)
plt.plot(t,rates_arr)
plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta')
plt.annotate('Theta', xy=(1.0, theta+0.0005))
plt.xlim(-0.05, 1.05)
plt.ylim([-0.05, 0.05])
plt.ylabel("Rate")
plt.xlabel("Time (yr)")
plt.title("Sigma scaled up 5 times")
plt.show()
