import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

# File paths 
filepath = "../Data/Compiled/The Pareto effect in tipping social networks Tipping Data - Tipping_Data.csv"

#%%


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

def preprocess_data(df):
    effect_magnitude_data = df[df['effect_magnitude'].notna()].copy()
    effect_cascade_data = df[df['effect_cascade_success'].notna()]
    effect_magnitude_data['numeric_magnitude'] = effect_magnitude_data['effect_magnitude'].apply(to_numeric_magnitude)
    return effect_magnitude_data, effect_cascade_data

def plot_effect_cascade_success(effect_cascade_data):
    effect_cascade_data['plotting_variable'] = effect_cascade_data['grouping_term'].combine_first(effect_cascade_data['variable'])
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 6))  # Adjusted the figure size to reduce white space
    
    effect_cascade_data = effect_cascade_data[effect_cascade_data['effect_cascade_success'] != '-/+']
    effect_cascade_data['colors'] = effect_cascade_data['effect_cascade_success'].map({'+': 'g', '-': 'r'}).fillna('b')
    
    size_data = effect_cascade_data.groupby('plotting_variable').size().reset_index(name='sizes')
    effect_cascade_data = effect_cascade_data.merge(size_data, on='plotting_variable', how='left')
    
    sns.scatterplot(x="effect_cascade_success", y="plotting_variable", size="sizes", marker='o',
                    sizes=(100, 500), hue='colors', palette={'g': 'g', 'r': 'r'}, data=effect_cascade_data, ax=ax, legend=None)
    
    ax.set(xlabel='Measure of effect', ylabel='')
    ax.set_title('Factors influencing cascade success')
    ax.set_xlim(-0.5, 1.5)  # Adjusted to reduce white space to the left of the "-" column
    plt.tight_layout()
    plt.savefig("../Figures/Cascade_success_final.png",dpi=600 )
    plt.show()
   


def main():
    # Load the data
    df = pd.read_csv(filepath)
    
    # Preprocess the data
    effect_magnitude_data, effect_cascade_data = preprocess_data(df)
    
    # Generate the modified "Effect Cascade Success" plot
    plot_effect_cascade_success(effect_cascade_data)
    
    # Add the "Magnitude and Cascade Size" plot (left unchanged, to be modified as per user's specific instructions)
    # ...

if __name__ == "__main__":
    main()
