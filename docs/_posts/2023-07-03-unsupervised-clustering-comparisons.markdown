---
layout: post
title:  "Article: K-Means is one way. But is it always the right way? What about K-Modes? Or Latent Class Analysis (LCA)?"
date:   2023-07-03 09:26:28 +0100
categories: medium article clustering
---

We’ve all done K-Means clustering examples and the results look good, but are they right, and what are other alternative Unsupervised Clustering Techniques

Kaggle, as we all know, is a wealth of data sets, do got this project I will use a Survey dataset from there. Specifically <a href="https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey" target="_blank">Mental Health in Tech Survey</a>.
I have included a copy of the data and all code in this <a href="https://github.com/mkeywood1/unsupervised-clustering-comparisons" target="_blank">github repo</a> for convenience.

### Overview

We’ll perform comparisons of outputs with multiple approaches.

Namely:
- K-Means (via one-hot encoding and scaling), PCA and Random Forest to identify Feature Importance
- K-Means (via LLM embeddings and scaling), PCA and Random Forest to identify Feature Importance
- K-Mode (via one-hot encoding), MCA and Random Forest to identify Feature Importance
- Latent Class Analysis (LCA)

### Notebooks
In my comparisons, I used a number of notebooks to test out the approaches. All are in the project repo.

### Run the app in Streamlit
For convenience I put everything into a simple Streamlit app too.

This is also in the project repo or a demo <a href="https://unsupervised-clustering-comparisons-mwk.streamlit.app" target="_blank">here</a>.

# Initial EDA and Feature Engineering on the Survey Data
So fundamentally, what are we trying to do?

Well, can we identify any natural groupings into personas?
Keep in mind that clustering is an exploratory technique and doesn’t necessarily imply causality. It should be used together with other methods to draw robust conclusions from survey responses.

That said, some questions that would likely yield insightful clusters from this survey include:

- Age, Gender, Country — This can give you basic demographic clusters, e.g. young females in the US, middle-aged males in the UK, etc.
- self_employed, tech_company, remote_work — This can cluster people into groups like tech startup employees, remote freelancers, traditional office employees, etc.
- family_history, treatment, seek_help — This can cluster people into groups like those with a family mental health history who have sought treatment, those with no family history who have still sought help, those with a family history who have not sought help, etc.
- work_interfere, mental_vs_physical, obs_consequence — This can cluster people into groups like those whose work is highly impacted by mental health, those more impacted by physical health, those whose work is not really impacted by health issues, etc.
- benefits, care_options, wellness_program — This can cluster companies/employees into groups like those with strong mental health benefits and support programs, those with moderate benefits and programs, those with little or no mental health support.
- anonymity, coworkers, supervisor — This can cluster people into groups based on how open and understanding their work environment is about mental health issues. E.g. open and accommodating environments, moderately open environments, non-open environments.
- Age, treatment, mental_health_interview — This can cluster people into groups based on their mental health diagnosis and treatment journey, e.g. those diagnosed and treated early in life, those diagnosed and treated later in life, those undiagnosed or untreated, etc.
- No employees, Remote work, Tech company, Benefits, Care options, Wellness program — Company size and culture
- Seek help, Anonymity, Leave, Mental health consequence, Physical health consequence, Coworkers, Supervisor, Mental health interview, Physical health interview, Mental vs physical, Obs consequence — Stigma and consequences
Combining columns across demographics, work life, health experiences and environment can provide very insightful clusters.

OK so let’s get on with it, with a bit of good old fashioned EDA.

The full code including details of decisions made can be found in `0 — initial_eda_and_feature_engineering.ipynb` but for brevity of the article the resultant functions for the EDA are:

{% highlight python %}
def age_ranges(x):
    "To be called through Pandas Apply, to map an Age integer to a binned string"
    res = "Unknown"
    if x < 18: res = "Adolescent"
    if x >= 18 and x < 40: res = "Adult"
    if x >= 40 and x < 60: res = "Middle aged"
    if x >= 60 and x < 70: res = "Senior citizens"
    if x >= 71: res = "Elderly"
    return res

def load_data(cols=[]):
    "Perform all the preprocessing steps we identified in the initial_eda_and_feature_engineering.ipynb notebook"
    df = pd.read_csv('survey.csv')

    # So it seems Gender was a free text field and as such is pretty noise. But every row has a value, which is good. There are some obvious fixes:
    # Male, male, M, m, Make, Male (with a space at the end), msle, Mail, Malr, maile, Mal, Cis Male, cis male, Cis Man, Male (CIS) are all 'M'
    # Female, female, F, f, Woman, Female (with a space at the end), femail, Femake, woman, Female (cis), cis-female/femme, Cis Female are all 'F'
    # For simplicity of this exercise I will group all others as 'Other'
    # NOTE: I am a strong supporter of LGBTQ+ rights and such please do not consider this simplification above as anything more for demonstration purposes for this project.
    for gender in ['Male', 'male', 'M', 'm', 'Make', 'Male ', 'msle', 'Mail', 'Malr', 'maile', 'Mal', 'Cis Male', 'cis male', 'Cis Man', 'Male (CIS)', 'Man']:
        df.loc[df['Gender'] == gender, 'Gender'] = 'Male'

    for gender in ['Female', 'female', 'F', 'f', 'Woman', 'Female ', 'femail', 'Femake', 'woman', 'Female (cis)', 'cis-female/femme', 'Cis Female']:
        df.loc[df['Gender'] == gender, 'Gender'] = 'Female'

    df['Gender'] = df['Gender'].apply(lambda x: 'Other' if x not in ['Male', 'Female'] else x)
    
    # A quick look at `comments` shows some interesting info there that could be used in a follow up experiment, but for now I will disregard the `comments` field.
    df.drop('comments', axis=1, inplace=True)

    # Let's look at 'state' first:
    # OK that largely makes sense, right, that the countries are non-US and so seemingly in this data are not using State or a State equivalent.
    # For simplicity lets fill the `nan` US ones with 'CA'.
    df.loc[(df['state'].isna()) & (df['Country'] == 'United States'), 'state'] = df['state'].mode()[0]

    # And set the rest (non-US) to 'N/A'
    df['state'] = df['state'].fillna("N/A")

    # Good, ok let's move on to `self employed`.
    # Interestingly the 18 records that do not have `self_employed` filled are the first 18 in the data_set, so maybe this was not asked fromt eh start.
    # Let's just set them to the mode of the `self_employed` column.
    df.loc[df['self_employed'].isna(), 'self_employed'] = df['self_employed'].mode()[0]

    # So finally, let's look at `work_interfere`.
    # It seems the middle of the road value of 'Sometimes' was the most answer, so let's just use that.
    df.loc[df['work_interfere'].isna(), 'work_interfere'] = df['work_interfere'].mode()[0]

    # Let's bin Age into something categorical.
    df['Age'] = df['Age'].apply(age_ranges)

    # Finally I think we can lose the `TimeStamp'.
    df.drop('Timestamp', axis=1, inplace=True)

    # Filter to just the selected columns
    if len(cols) > 0: df = df[cols]
    
    # separate continuous and categorical variable columns
    # (Although this is boilerplate I use and not really relevant as we have binned the only numerical column ('Age'))
    continuous_vars = [col for col in df.columns if df[col].dtype != 'object']
    categorical_vars = [col for col in df.columns if df[col].dtype == 'object']
    
    if len(continuous_vars) > 0:
        # Scaling is important for K-Means because K-Means is a distance-based algorithm that clusters data points based on their Euclidean distance from a centroid. If the features in the dataset are not scaled, some of them may be given higher weights than others, which can result in clustering biases towards features with larger magnitudes. This can lead to poor cluster assignments and reduced accuracy
        scaler = MinMaxScaler()
        df_con[continuous_vars] = pd.DataFrame(scaler.fit_transform(df[continuous_vars]))
    else:
        df_con = pd.DataFrame()
    
    if len(categorical_vars) > 0:
        df_cat = pd.get_dummies(df, columns=categorical_vars)
    else:
        df_cat = pd.DataFrame()
        
    df_preprocessed = pd.concat([df_con, df_cat], axis=1)
    
    return df, df_preprocessed
{% endhighlight %}

Great — This is ready to be used in the other Notebooks.

# K-Means Clustering
The full code can be found in `1 — kmeans_clustering_inc_using_LLM.ipynb` but for brevity of the article I’m just pulling out interesting pieces:

So let’s define some other helpful functions. These will be used to show the data for the relevant clusters, show the scatter plot to visualize the clusters, and attempt to identify the most important question/answer pairs for the algorithm.

{% highlight python %}
def show_clusters(df_pca, df, col='predicted_cluster'):
    "Simply output the contents of the dataframe per Cluster to see those grouped together"
    for cluster_number in range(len(df_pca[col].unique())):

        idx = df_pca[df_pca[col] == cluster_number].index

        print('Cluster', str(cluster_number))
        display(df.iloc[idx])
        print('---------------')



def show_scatter(df_pca):
    "Output a scatter plot of the PCA'd (2 dimensions) data"
    plt.figure(figsize=(8, 8))

    scat = sns.scatterplot(
        x="component_1",
        y="component_2",
        s=50,
        data=df_pca,
        hue="predicted_cluster",
        palette="Set1",
    )

    # loop through the data and add annotations for each point
    for i in range(len(df_pca)):
        label = int(i)  # get the label from the 'index' column in the data
        x = df_pca.iloc[i]['component_1']  # get the x-coordinate
        y = df_pca.iloc[i]['component_2']  # get the y-coordinate
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    scat.set_title("Clustering results")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.show()


def get_important_features(df_preprocessed):
    """
    In an attempt to get the most important features identified, let's use a Random Forest classifier.
    We will pass in the training data with the proposed clusters as the labels, which when the model is trained
    the feature importance should be insightful as to how it trained to those labels aka Clusters.
    """
    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_preprocessed.drop('predicted_cluster', axis=1), df_preprocessed['predicted_cluster'])

    # Get feature importances
    importances = rf.feature_importances_

    # Sort feature importances in descending order
    sorted_indices = importances.argsort()[::-1]

    # Get sorted feature names and importances
    sorted_features = [df_preprocessed.columns[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    # Plot feature importances as horizontal bar chart
    plt.barh(range(10), sorted_importances[:10], align='center')
    plt.yticks(range(10), sorted_features[:10])
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.show()
{% endhighlight %}

## First try K-Means with One Hot Encoding and Normalization
First load the data and limit to the questions we want. We create appropriate pipeline for the preprocessing. In this case it will be PCA.
Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to reduce the number of features in a dataset, while retaining as much of the original variance and information in the dataset as possible. Therefore PCA can be useful for preprocessing data before clustering.

{% highlight python %}
df, df_preprocessed = load_data()
preprocessor_pca = Pipeline([
       (“pca”, PCA(n_components=2, random_state=42)),
   ])

preprocessed_X = preprocessor_pca.fit_transform(df_preprocessed)
preprocessed_X
{% endhighlight %}

But how can we determine the right number of clusters?
There are two good methods we can use to try and determine the appropriate value for k. The **elbow method** and **silhouette method**.
The elbow method involves plotting the relationship between the number of clusters (k) and the sum of squared distances between data points and their assigned cluster center. The ideal k value is the point where the decrease in sum of squared distances starts to level off (i.e., the elbow point)
The silhouette method involves calculating the average silhouette score for each number of clusters k. The silhouette score measures how similar a data point is to its assigned cluster compared to other clusters. It ranges from -1 to 1, where values closer to 1 indicate that a data point is well matched to its cluster.

{% highlight python %}
# Elbow method to select k
# The ‘elbow’ of the curve indicates the best number of clusters.

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 1972,
}

# A list holds the sum of squared errors (SSE) values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(preprocessed_X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/1.png">

{% highlight python %}
# Silhouette method to select k
# The highest point indicates the best number of clusters.

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice we start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(preprocessed_X)
    score = silhouette_score(preprocessed_X, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/2.png">

Both the Elbow and Silhouette make it look like **4** is a good choice for the number of clusters.

{% highlight python %}
# Set the number of clusters to be the value identified from the silhouette score
n_clusters = silhouette_coefficients.index(max(silhouette_coefficients)) + 2

print(f"{n_clusters}")

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=n_clusters,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor_pca", preprocessor_pca),
        ("clusterer", clusterer)
    ]
)

_ = pipe.fit(df_preprocessed)
{% endhighlight %}

So let’s just check the Silhouette score again.

{% highlight python %}
predicted_labels = pipe["clusterer"]["kmeans"].labels_
silhouette_score(preprocessed_X, predicted_labels)

# 0.38442501822601755
{% endhighlight %}

As Silhouette ranges from -1 to 1, this is not bad.
Finally let’s look at these as clusters.

{% highlight python %}
df_pca = pd.DataFrame(
    pipe["preprocessor_pca"].transform(df_preprocessed),
    columns=["component_1", "component_2"],
)

# Add in the cluster
df_pca["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
df_preprocessed['predicted_cluster'] = df_pca['predicted_cluster']

show_scatter(df_pca)
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/3.png">

`show_clusters(df_pca, df_preprocessed)`

This gives us each cluster and the records in them in turn.

Finally, let’s see if we can see what features influenced it the most.

`get_important_features(df_preprocessed)`

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/4.png">

OK so that’s looking pretty good, but can we use something other than the fundamental one hot embeddings? How about using the LLM’s for the Embeddings?

## Using LLM’s for embeddings rather than One Hot Encodings and Normalization
What we are going to do here is:

- Build up a long string of a concatenation of all the raw text results for each survey result
- Use SentenceTransformer to get the embeddings for this long string
- Use these embeddings as the processed data to be passed into the K-Means model

A lot of this is similar to the last bit except our pipeline.

{% highlight python %}
df, df_preprocessed = load_data()

sbert_model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")

def compile_text(df):
    res = {}
    for idx, row in df.iterrows():
        text = ""
        for col in df.columns:
            if row[col] != "": 
                text += f'{col}: "{row[col]}"\n'
#         print(text)
#         print("-----------")
        res[idx] = output_embedding(text.strip()).values[0]
    return pd.DataFrame(res).T

def output_embedding(txt):
    embd = sbert_model.encode(txt)
    return pd.DataFrame(embd.reshape(-1, 384))

# Get the LLM Embeddings
df_embed = compile_text(df)

# Still scale it though, to keep K-Means happy, and push through PCA
preprocessor_2 = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

preprocessed_X = preprocessor_2.fit_transform(df_embed)
{% endhighlight %}

We then do the Elbow, Silhouette, etc as per the last example, resulting in the following clustering and important features:

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/5.png">

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/6.png">

OK so this seems very very different? Not wrong, just different.

I think there is some interesting ideas in using LLM embeddings here, that I will come back to in another article.

# Hmmmmm, have we gone wrong somewhere
Have we used the right algorithm?

K-means is used for clustering numerical data, whereas realistically this whole dataset was treated with binning and one hot encoding and so is categorical.

K-modes should be used for clustering categorical data.

Or in more detail:

- K-means is a partition-based clustering algorithm that aims to partition n instances into k clusters, such that instances within the same cluster are as similar as possible, while instances from different clusters are as dissimilar as possible. The algorithm works by iteratively assigning instances to the cluster with the closest centroid (average position) and updating the centroid based on the mean of the assigned instances. K-means is commonly used for clustering numerical data, such as continuous variables
- K-modes is an extension of k-means that is designed for clustering categorical data. The algorithm replaces the distance metric with a dissimilarity measure that takes into account the categorical nature of the data. Instead of computing the Euclidean distance between instances, K-modes uses a simple matching measure that counts the number of mismatched attributes between instances. The centroids are also represented using the mode (most common value) instead of the mean

# So let’s move on to another notebook where we explore K-modes

The full code can be found in `2 — kmodes_and_mca_example.ipynb` but for brevity of the article I’m just pulling out interesting pieces.

The EDA, helper functions etc are the same, so let’s start with the Elbow calculations.

{% highlight python %}
cost = []
K = range(1,7)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init="random", n_init=5)
    kmode.fit_predict(df_preprocessed)
    cost.append(kmode.cost_)
    
plt.plot(K, cost)
plt.xlabel("Number of Clusters")
plt.ylabel('Cost')
plt.show()
{% endhighlight %}

Then find the elbow (where the slope declines and it begins to level off).

To determine the elbow point programatically, you can analyze the rate of change in the slope of the curve. One approach is to calculate the second derivative of the curve, which measures the curvature of the curve at each point. The elbow point can be determined as the point with the maximum distance from the line connecting the first and last points of the curve.

{% highlight python %}
# Calculate the second derivative of the curve
cost = np.array(cost)
diff1 = np.diff(cost)
diff2 = np.diff(diff1)
diff2 = np.insert(diff2, 0, 0)

optimal_K = np.argmax(diff2) + 1
print(f"{optimal_K}")
{% endhighlight %}

This gives an optimal K value, so let’s build with that.

{% highlight python %}
# Building the model with 'optimal_K' clusters
kmode = KModes(n_clusters=optimal_K, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(df_preprocessed)

df_preprocessed.insert(0, "predicted_cluster", clusters, True)
{% endhighlight %}

OK so now can we visualize the clusters with MCA.

{% highlight python %}
# Perform MCA
df_mca = mca.MCA(df_preprocessed, ncols=3)
mca_results = df_mca.fs_r(N=len(df_preprocessed))

# View the eigenvalues 
# print(df_mca.L) 

# Plot the results 
plt.scatter(mca_results[:,0], mca_results[:,1], c=df_preprocessed['predicted_cluster'].to_list())
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

for label, x, y in zip(df_preprocessed.index, mca_results[:,0], mca_results[:,1]):
    plt.annotate(label, xy=(x, y), xytext=(x + .02, y))

plt.show()
{% endhighlight %}

Honestly, I was so underwhelmed with the results here that I will spare you the visuals, but please feel free to look in the notebook in the project repo.

Disappointing :-(

After a lot of experimenting with K-Modes I couldn’t get near what I wanted, so let’s try something else.

# Let’s try Latent Class Analysis in the next Notebook
The full code can be found in `3 — latent_class_analysis_(lca).ipynb` but for brevity of the article I’m just pulling out interesting pieces.

The EDA, helper functions etc mostly the same, with just one addition, that I will come to shortly.

LCA, simply put, is a statistical technique used to identify subgroups (or latent classes) within a larger population, based on patterns of responses to a set of categorical variables.

LCA assumes that the underlying latent classes are unobservable, and their membership can only be inferred based on observed responses to various items.

The LCA models aim to identify the optimal number of latent classes, and to estimate the probability of each individual belonging to each class, given his/her response pattern. The models generate a set of estimated probabilities for each person’s membership in each class.

So let’s give it a go…

Let’s start with a guess at 3 classes (components) for 1 iteration (step).

{% highlight python %}
model = StepMix(n_components=3, n_steps=1, measurement="categorical", verbose=1, progress_bar=0, random_state=42)

# Fit model and predict clusters
model.fit(df_preprocessed)
clusters = model.predict(df_preprocessed)
clusters
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/7.png">

OK so that has done something, but can we see what?

{% highlight python %}
# What are the features used and parameters generated?
model.feature_names_in_, model.get_parameters()
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/8.png">


After a bit of reading around I can see that it’s every other parameter I want for the y-axis

`np.array(list(model.get_parameters()['measurement']['pis'][0][::2]))`

So this can give me something like this, for a single class:

{% highlight python %}
def show_lca(model, n_clusters, classes=[]):
    # Plot the results 
    fig = plt.figure(figsize=(8, 8))

    x_range = range(int(model.get_parameters()['measurement']['total_outcomes']  / 2))

    # for i, values in enumerate(model.get_parameters()['measurement']['pis']):
        # plt.plot(x_range, np.array(list(values)[::2]), label=f'Class {i}')

    if len(classes) == 0: classes = ['Class '+str(i) for i in range(n_clusters)]
    for cls in classes: #:
        plt.plot(x_range, np.array(list(model.get_parameters()['measurement']['pis'][int(cls[-1])][::2])), label=cls)

    # Add grid lines 
    plt.grid(True)  

    # Add a legend
    plt.legend()  

    # Add x-axis and y-axis labels
    plt.xlabel('Question and Answer')
    plt.ylabel('Question Relevancy') 

    # Add the ticks
    plt.xticks(range(len(model.feature_names_in_)), model.feature_names_in_, rotation=90, fontsize=8)
    plt.yticks([0, 1], ['Relevant', 'Not Relevant'])

    plt.show()


show_lca(model, 3, ['Class 0'])
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/9.png">

Awesome, so we can see the line ‘point to’ the question / answer pair that influenced it. In this case, Class 0 is influenced by Adult Males.

So let’s put all that together, including a Grid Search to find some best parameters.

We will use the AIC (Akaike Information Criterion). It is a widely used measure of goodness of fit for statistical models, like LCA, that attempts to balance model fit with model parsimony. Specifically, the AIC score for a model represents a trade-off between the model’s deviation from the data and the number of parameters used to explain the data.

{% highlight python %}
# Scikit-Learn grid search object. We test n_classes from 1 to 8.
# We also add 1-step, 2-step and 3-step to the grid.
# We use 3 validation folds.
# We therefore fit a total of  8x3x3=72 estimators.
grid = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8],
    'n_steps' : [1, 2, 3]
}

model = StepMix(n_components=3, n_steps=1, measurement='categorical', verbose=0, progress_bar=0, random_state=123)

results = dict(param_n_steps=[], param_n_components=[], aic=[], bic=[])

# Same model and grid as above
for g in ParameterGrid(grid):
    model.set_params(**g)
    model.fit(df_preprocessed)
    results['param_n_steps'].append(g['n_steps'])
    results['param_n_components'].append(g['n_components'])
    results['aic'].append(model.aic(df_preprocessed))
    results['bic'].append(model.bic(df_preprocessed))

# Save results to a dataframe
results = pd.DataFrame(results)

n_clusters = int(results.loc[results['aic'].idxmin()]['param_n_components'])
n_steps = int(results.loc[results['aic'].idxmin()]['param_n_steps'])

print(f'### Using AIC determined best Cluster Size of {n_clusters}')

# Categorical StepMix Model with n_clusters latent classes
model = StepMix(n_components=n_clusters, n_steps=n_steps, measurement="categorical", verbose=0, progress_bar=0, random_state=123)
# Fit model and predict clusters
model.fit(df_preprocessed)
clusters = model.predict(df_preprocessed)

print(model.get_parameters())

df_preprocessed['predicted_cluster'] = clusters
df_ca = pd.DataFrame({'predicted_cluster': clusters})

print('Show the LCA results')
show_lca(model, n_clusters)

print('Show the Cluster Data')
show_clusters(df_ca, df_preprocessed) 
{% endhighlight %}

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/10.png">

Fantastic !!!

We can see the questions / answers that were relevant to each cluster, including the degree of relevance.

Next step pull the whole lot together in a Streamlit App called ‘streamlit_clustering.py’, also in the repo. In this you can try the different ones, which fields to use, which algorithms to use, which clusters to visualize etc.

<img src="https://mkeywood1.github.io/2023-07-03-unsupervised-clustering-comparisons/11.png">

Overall, for me, in this case and what I wanted to achieve, LCA won hands down!!!

Thanks for reading :-)
