# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h2> GOOGLE PLAYSTORE DATASET ANALYSIS </h2>
# <li> <b>Dataset:</b> https://www.kaggle.com/lava18/google-play-store-apps *as on 16.09.2020*</li> 
# <li> <b>Goal:</b> Perform exploratory data analysis and visualizations </li>
# <li> <b>List of Questions: </b> </li> <br>
# <ol> 1. Distribution of free and paid apps by category </ol>
# <ol> 2. Popularity of free and paid apps by category </ol>
#    

# %%
# Library imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
# import dataset

gps_ds = pd.read_csv(r"~\googleplaystore_analysis\data\raw\googleplaystore.csv") #edit this
gps_ds.head(5)

# %% [markdown]
# ## Data Cleaning 
# #### Step 1: Check for datatype incompatibilities
# 
# **Inferences:** We beed to cast some of the fields as follows: 
# - 'Price' --> float
# - 'Last-Updated' --> datetime
# - 'Rating' --> float
# - 'Reviews' --> Integer
# - 'Size' and 'Installs' --> --tba--

# %%
#gps_ds.info()

# dropping one row from the dataset which is of poor quality before datatype casting
gps_ds = gps_ds[gps_ds.App != "Life Made WI-Fi Touchscreen Photo Frame"]


# %%
# Column type casting

gps_ds["Reviews"] = pd.to_numeric(gps_ds["Reviews"])
gps_ds["Rating"] = pd.to_numeric(gps_ds["Rating"])

# remove the currency code and then cast to numeric
gps_ds["Price"] = gps_ds["Price"].map(lambda x:x.lstrip('$'))
gps_ds["Price"] = pd.to_numeric(gps_ds["Price"])

gps_ds["Last Updated"] = pd.to_datetime(gps_ds["Last Updated"])

# remove the symbols + and , and then cast to numeric
gps_ds["Installs"] = gps_ds["Installs"].map(lambda x:x.rstrip('+'))
gps_ds["Installs"] = gps_ds["Installs"].str.replace(',','')
gps_ds["Installs"] = pd.to_numeric(gps_ds["Installs"])
#################################### end of casting ####################################

#gps_ds.info()

# %% [markdown]
# #### Step 2: NaN(null) value handling :-
# 

# %%
# Step 2.1: Let's check the NaN stats(count) across the dataframe first
print(gps_ds.isnull().sum())

# Step 2.2: We can infer that, there is about 13.59% records with missing values for the 'Rating' attribute. We can impute this with 0s because:
# It is safe to assume that apps whose reviews are filled with NaN could mean that there has been no reviews yet. Since the other records have count of reviews in this dataset, let us replace NaN with 0 safely and avoid losing these records otherwise.

gps_ds["Reviews"] = gps_ds["Reviews"].fillna(0)

# %% [markdown]
# #### Preliminery data analysis for understanding
# 

# %%
print("1. Most popular categories in terms of no. of apps:\n")
print(gps_ds.groupby(["Category","Type"])["Category"].count().sort_values(ascending=False).head(10))

# %% [markdown]
# ## DATA ANALYSIS AND VISUALIZATION
# ### 1. Distribution of free and paid apps by category:

# %%
## Let us visualise the distribution of apps as a stacked bar chart were blue and orange colors denote Free and Paid apps respectively.
apps_by_category = gps_ds.groupby(["Category","Type"])["App"].count()
apps_by_category = apps_by_category.unstack()
print('\n')
apps_by_category.plot(kind='bar',stacked=True,figsize=(10,5))
plt.title("Distribution of free and paid apps by category")
plt.xlabel("Category")
plt.ylabel("No. of Apps")
plt.show()

# %% [markdown]
# _'Family','Game' and 'Tools' seem to be the top 3 popular category in both Paid and Free apps_
# %% [markdown]
# ### 2. Distribution of free and paid apps by category and no. of downloads:

# %%
gps_ds.head(10)


# %%
# Let us convert the no. of Installs into categorical data for a better summary
buckets = pd.cut(gps_ds.Installs,bins=[0,1000,100000,100000000,10000000000],labels=['0-1000','1001-100000','100001-100000000','100000001-10000000000'])
gps_ds.insert(5,"downloads_range",buckets)


# %%
# splitting the main df into two:
apps_by_download_free = gps_ds.loc[gps_ds['Type'] == "Free"]
apps_by_download_paid = gps_ds.loc[gps_ds['Type'] == "Paid"]


# %%
# Create a subplot of dimension 1 row X 2 columns
installs_free = apps_by_download_free.groupby(["downloads_range","Type"])["Type"].count()
installs_paid = apps_by_download_paid.groupby(["downloads_range","Type"])["Type"].count()

fig = plt.figure()

# cell 1x1 of a 1x2 grid, for Free Apps category and cell 1x2 for Paid Apps:

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

installs_free.plot(kind='bar',x='downloads_range', ax=ax1, legend=False,title="Free Apps",figsize=(10,6))
installs_paid.plot(kind='bar',x='downloads_range', ax=ax2, legend=False,title="Paid Apps",figsize=(10,6))

ax1.set(xlabel='total downloads bucket', ylabel='No. of Installs')
ax2.set(xlabel='total downloads bucket', ylabel='No. of Installs')

plt.show()
fig2 = plt.show()

#df.groupby('country')['unemployment'].mean().sort_values().plot(kind='barh', ax=ax2)

# %% [markdown]
# _ We can see that, only a few number of free apps are extremely successful. and not many download paid apps in comparison_ 
# %% [markdown]
# ### 3. Effect of Price and size in the no. of downloads of Paid Apps:

# %%
apps_by_download_paid = apps_by_download_paid.drop_duplicates(keep='first')


# %%
apps_by_download_paid.head(5)

# %% [markdown]
# ### 4. Are apps worth buying? (analysis based on average user rating)

# %%
apps_by_download_paid.head(5)


# %%

df1 = apps_by_download_paid.groupby(["Category"])["Rating"].median().sort_values(ascending=False).to_frame()
df2 = apps_by_download_free.groupby(["Category"])["Rating"].median().sort_values(ascending=False).to_frame()
rating_comparison = df1.join(df2,on='Category',lsuffix='_paid_apps',rsuffix='_free_apps').sort_values('Category')

fig3 = rating_comparison.plot(kind='bar',y=['Rating_paid_apps','Rating_free_apps'],title='Performance Free vs. Paid Apps',figsize=(20,10))

# %% [markdown]
# _looks like paid apps perform marginally better than the free apps_

