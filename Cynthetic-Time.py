import numpy as np
import pandas as pd

def generate_multivariate_time_series(T, M, patterns, correlation_function, heteroscedasticity):
    data = np.zeros((T, M))
    
    for m in range(M):
        pattern = patterns[m % len(patterns)](T)
        if heteroscedasticity:
            pattern *= np.random.uniform(0.5, 1.5, T)
        data[:, m] = pattern
        
    correlations = correlation_function(M)
    correlated_data = np.dot(data, correlations)
    return correlated_data

def correlation_function(M):
    correlations = np.random.rand(M, M)
    correlations = (correlations + correlations.T) / 2  # Make the matrix symmetric
    np.fill_diagonal(correlations, 1)  # Set diagonal values to 1
    return correlations

def generate_synthetic_data(N, T, M, K, patterns, noise_std, P_miss, non_temporal_vars, missing_mechanism, correlation_function, heteroscedasticity):
    data = []
    non_temporal_data = []

    samples_per_pattern = N // K
    extra_samples = N % K

    for k in range(K):
        base_ts = generate_multivariate_time_series(T, M, patterns, correlation_function, heteroscedasticity)
        samples = samples_per_pattern + (1 if k < extra_samples else 0)
        
        for _ in range(samples):
            ts = base_ts + np.random.normal(0, noise_std, (T, M))
            non_temporal_sample = np.random.normal(ts.mean(axis=0)[:non_temporal_vars], noise_std)
            
            # Apply the missingness mechanism
            ts, non_temporal_sample = apply_missingness(ts, non_temporal_sample, P_miss, missing_mechanism)

            data.append(ts)
            non_temporal_data.append(non_temporal_sample)

    data = np.stack(data, axis=0)
    non_temporal_data = np.stack(non_temporal_data, axis=0)

    return data, non_temporal_data


def apply_missingness(ts, non_temporal_sample, P_miss, missing_mechanism):
    T, M = ts.shape

    if missing_mechanism == 'MCAR':
        missing_indices = np.random.choice(T * M, int(T * M * P_miss), replace=False)
        for idx in missing_indices:
            ts[idx // M, idx % M] = np.nan
            
    elif missing_mechanism == 'MAR':
        for m in range(M):
            P_miss_col = P_miss * (1 + ts[:, m].mean())  # Example: Increase the probability of missingness based on the column mean
            missing_indices = np.random.choice(T, int(T * P_miss_col), replace=False)
            ts[missing_indices, m] = np.nan

        for idx in range(len(non_temporal_sample)):
            P_miss_col = P_miss * (1 + non_temporal_sample.mean())  # Example: Increase the probability of missingness based on the mean
            if np.random.rand() < P_miss_col:
                non_temporal_sample[idx] = np.nan

    elif missing_mechanism == 'MNAR':
        for m in range(M):
            missing_indices = np.where(ts[:, m] > np.percentile(ts[:, m], 100 - 100 * P_miss))[0]  # Example: Set missing values for the top P_miss percent values
            ts[missing_indices, m] = np.nan

        for idx in range(len(non_temporal_sample)):
            if non_temporal_sample[idx] > np.percentile(non_temporal_sample, 100 - 100 * P_miss):  # Example: Set missing values for the top P_miss percent values
                non_temporal_sample[idx] = np.nan

    else:
        raise ValueError("Invalid missingness mechanism specified.")

    return ts, non_temporal_sample

# Define different patters for time-series variables
def pattern1(T):
    x = np.linspace(0, 2 * np.pi, T)
    return np.sin(x)

def pattern2(T):
    x = np.linspace(0, 2 * np.pi, T)
    return np.cos(x)

def pattern3(T):
    x = np.linspace(0, 2 * np.pi, T)
    return np.sin(2 * x)

def pattern4(T):
    x = np.linspace(0, 2 * np.pi, T)
    return np.cos(2 * x)

def pattern5(T):
    x = np.linspace(0, 2 * np.pi, T)
    return np.sin(x) * np.cos(x)

patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]


# In[1]:


#Example
# Set the parameters for generating synthetic data
N = 1000  # Number of samples
T = 100    # Length of each time series
M =50      # Number of variables in each multivariate time series
K = 3     # Number of distinct patterns
noise_std = 0.1  # Noise standard deviation
P_miss = 0.1  # Probability of missing values
non_temporal_vars = 10  # Number of non-temporal variables
missing_mechanism = 'MAR'  # Missingness mechanism: 'MCAR', 'MAR', or 'MNAR'
heteroscedasticity = True  # Enable heteroscedasticity

# Generate synthetic data
data, non_temporal_data = generate_synthetic_data(N, T, M, K, patterns, noise_std, P_miss, non_temporal_vars, missing_mechanism, correlation_function, heteroscedasticity)

# Convert the generated data to pandas DataFrames
time_series_data = [pd.DataFrame(data[i], columns=[f'var_{j}' for j in range(M)]) for i in range(N)]
non_temporal_data = pd.DataFrame(non_temporal_data, columns=[f'non_temporal_var_{j}' for j in range(non_temporal_vars)])

# Display a sample time series
print("Sample time series data (first 10 rows):")
print(time_series_data[0].head(10))

# Display non-temporal data
print("\nSample non-temporal data (first 10 rows):")
print(non_temporal_data.head(10))

