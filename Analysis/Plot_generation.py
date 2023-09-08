import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

# File paths (these should be updated based on the actual paths in your system)
filepath = "./Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data.csv"


#%%
# Read the data
df = pd.read_csv(filepath)


# Convert the 'effect_magnitude' values to numerical form to create a 'numeric_magnitude' column
def to_numeric_magnitude(value):
    try:
        return float(value)
    except:
        if value == '+':
            return 1
        elif value == '-':
            return -1
        else:
            return None
        
        
# Filter rows where 'effect_magnitude' and 'effect_cascade_success' are not NaN
effect_magnitude_data = df[df['effect_magnitude'].notna()].copy()
effect_cascade_data = df[df['effect_cascade_success'].notna()]

# Add a new column with numerical magnitude values
effect_magnitude_data['numeric_magnitude'] = effect_magnitude_data['effect_magnitude'].apply(to_numeric_magnitude)

# Group the data for the "Effect Magnitude" bar plot
effect_magnitude_data_grouped = effect_magnitude_data.groupby(['variable', 'numeric_magnitude']).size().reset_index(name='ref_count')

# Group the data for the "Effect Cascade Success" scatter plot and filter for 'ref_count' >= 1
effect_cascade_data_grouped = effect_cascade_data.groupby(['variable', 'effect_cascade_success']).size().reset_index(name='ref_count')
effect_cascade_data_filtered = effect_cascade_data_grouped[effect_cascade_data_grouped['ref_count'] >= 1]

#%%
# Generate the bar plot for "Effect Magnitude"
plt.figure(figsize=(12, 8))

sns.barplot(data=effect_magnitude_data_grouped.dropna(subset=['numeric_magnitude']), 
            x='variable', y='numeric_magnitude', hue='ref_count', dodge=False, 
            palette='viridis', edgecolor=(0, 0, 0, 0))

variables_with_range = effect_magnitude_data[effect_magnitude_data['effect_magnitude'].str.contains('-/\+', na=False)]['variable'].unique()
for idx, var in enumerate(variables_with_range):
    if var in effect_magnitude_data_grouped['variable'].unique():
        plt_idx = list(effect_magnitude_data_grouped['variable'].unique()).index(var)
        plt.plot([plt_idx, plt_idx], [-3, 3], color='gray', lw=1.5)
        plt.plot([plt_idx - 0.15, plt_idx + 0.15], [3, 3], color='gray', lw=1.5)
        plt.plot([plt_idx - 0.15, plt_idx + 0.15], [-3, -3], color='gray', lw=1.5)

plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
plt.title('Effect Magnitude vs Variable')
plt.ylabel('Effect Magnitude')
plt.xlabel('Variable')
plt.legend(title='N', loc='upper right')
plt.tight_layout()
plt.show()


#%%
# Generate the updated scatter plot for "Effect Cascade Success"
plt.figure(figsize=(12, 8))
sns.scatterplot(data=effect_cascade_data_filtered, x='effect_cascade_success', y='variable', 
                hue=effect_cascade_data_filtered['effect_cascade_success'].apply(lambda x: 'Negative' if x == '-' else 'Others'),
                size='ref_count', sizes=(40, 400), alpha=0.7, legend=True, marker='o')
plt.xticks(fontsize=12)
plt.xlim(-1.5, 1.5)
plt.ylim(-1, len(effect_cascade_data_filtered['variable'].unique()))
plt.xlabel('Effect Cascade Success', labelpad=20, fontsize=12)
plt.title('Variable vs Effect Cascade Success')
plt.ylabel('Variable')
plt.tight_layout()
plt.show()
