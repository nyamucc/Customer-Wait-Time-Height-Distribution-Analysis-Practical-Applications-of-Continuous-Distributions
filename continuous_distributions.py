# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# %% [markdown]
#  Generate Synthetic Customer Wait Times (Exponential Distribution)

# %%
np.random.seed(42)

mean_wait_time = 5  
scale = mean_wait_time  

# Generate synthetic wait times (1000 customers)
wait_times = np.random.exponential(scale, 1000)


# %%
mean_height = 170  
std_height = 10    
heights = np.random.normal(mean_height, std_height, 1000)  # Generate 1000 synthetic heights

# %%

exp_params = stats.expon.fit(wait_times, floc=0)  # Fit Exponential distribution (location fixed to 0)

norm_params = stats.norm.fit(heights)  # Fit Normal distribution


# %%

x_wait = np.linspace(0, max(wait_times), 1000)
x_height = np.linspace(min(heights), max(heights), 1000)
exp_pdf = stats.expon.pdf(x_wait, *exp_params)

plt.figure(figsize=(12, 6))

# Plot the histogram of wait times (Exponential Data)
plt.subplot(1, 2, 1)
plt.hist(wait_times, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Wait Times (Exponential)')
plt.plot(x_wait, exp_pdf, 'r-', label='Fitted Exponential Distribution')
plt.title('Exponential Distribution (Customer Wait Times)')
plt.xlabel('Wait Time (minutes)')
plt.ylabel('Density')
plt.legend()


# %%
norm_pdf = stats.norm.pdf(x_height, *norm_params)  # Normal PDF
# Plot the histogram of heights (Normal Data)
plt.subplot(1, 2, 2)
plt.hist(heights, bins=30, density=True, color='lightgreen', edgecolor='black', alpha=0.7, label='Heights (Normal)')
plt.plot(x_height, norm_pdf, 'g-', label='Fitted Normal Distribution')
plt.title('Normal Distribution (Heights)')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()


# %%
# Calculate the statistical properties for both datasets

# Exponential Distribution (Wait Times)
exp_mean = np.mean(wait_times)
exp_var = np.var(wait_times)
exp_skew = stats.skew(wait_times)

# Normal Distribution (Heights)
norm_mean = np.mean(heights)
norm_var = np.var(heights)
norm_skew = stats.skew(heights)

# Results
print("**Statistical Comparison of Exponential and Normal Distributions**\n")
print(f"Exponential Distribution (Wait Times):")
print(f"  Mean:         {exp_mean:.2f}")
print(f"  Variance:     {exp_var:.2f}")
print(f"  Skewness:     {exp_skew:.2f}\n")

print(f"Normal Distribution (Heights):")
print(f"  Mean:         {norm_mean:.2f}")
print(f"  Variance:     {norm_var:.2f}")
print(f"  Skewness:     {norm_skew:.2f}")


# %%
# Test for Exponential distribution (Wait Times)
ks_stat, ks_p_value = stats.kstest(wait_times, 'expon', args=(0, np.mean(wait_times)))

# Test for Normal distribution (Wait Times)
shapiro_stat, shapiro_p_value = stats.shapiro(wait_times)

# Test for Exponential distribution (Heights)
ks_stat_height, ks_p_value_height = stats.kstest(heights, 'expon', args=(0, np.mean(heights)))

# Test for Normal distribution (Heights)
shapiro_stat_height, shapiro_p_value_height = stats.shapiro(heights)

print("\n*** Distribution Test Results ***\n")


print(f"1. **Wait Times - Exponential Distribution Test (KS Test)**")
print(f"   - KS Statistic: {ks_stat:.3f} | P-value: {ks_p_value:.3f}")
print(f"   - Conclusion: {'Likely Exponential' if ks_p_value > 0.05 else 'Does not follow Exponential'}\n")


print(f"2. **Wait Times - Normal Distribution Test (Shapiro-Wilk Test)**")
print(f"   - Shapiro Statistic: {shapiro_stat:.3f} | P-value: {shapiro_p_value:.3f}")
print(f"   - Conclusion: {'Likely Normal' if shapiro_p_value > 0.05 else 'Does not follow Normal'}\n")


print(f"3. **Heights - Exponential Distribution Test (KS Test)**")
print(f"   - KS Statistic: {ks_stat_height:.3f} | P-value: {ks_p_value_height:.3f}")
print(f"   - Conclusion: {'Likely Exponential' if ks_p_value_height > 0.05 else 'Does not follow Exponential'}\n")

print(f"4. **Heights - Normal Distribution Test (Shapiro-Wilk Test)**")
print(f"   - Shapiro Statistic: {shapiro_stat_height:.3f} | P-value: {shapiro_p_value_height:.3f}")
print(f"   - Conclusion: {'Likely Normal' if shapiro_p_value_height > 0.05 else 'Does not follow Normal'}\n")


