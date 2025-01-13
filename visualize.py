import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(data):
    print('\n Visualizing data... be patient, this may take a while \n')
    sns.plotting_context('notebook', font_scale=1.2)

    columns_to_plot = ['sqft_lot', 'sqft_above', 'price', 'sqft_living', 'bedrooms', 'grade', 'yr_built', 'yr_renovated']
    g = sns.pairplot(data[columns_to_plot], hue='bedrooms', height=2)
    g.set(xticklabels=[])
    print('Finished creating pair plot')

    # Individual plots
    sns.jointplot(x='sqft_lot',y='price',data=data,kind='reg',height=4)
    sns.jointplot(x='sqft_above',y='price',data=data,kind='reg',height=4)
    sns.jointplot(x='sqft_living',y='price',data=data,kind='reg',height=4)
    sns.jointplot(x='yr_built',y='price',data=data,kind='reg',height=4)
    print('Finished creating individual plots')

    # Correlation heatmap
    print('Finished creating correlation heatmap \n')
    plt.figure(figsize=(15,10))
    columns =['price','bedrooms','bathrooms','sqft_living','floors','grade','yr_built','condition']
    sns.heatmap(data[columns].corr(),annot=True)
    print('Finished creating all visualizations \n')
    plt.show()