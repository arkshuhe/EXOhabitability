import os
import pandas as pd

def extract_tweets_from_files(folders_path='./00.assets/data/source'):
    """
    Extracts tweet data from TSV files and returns a DataFrame.
    
    Parameters:
        folders_path: path to store the tsv files
    Returns:
      A pandas DataFrame containing the extracted data.
    """

    # Define the columns you want to extract
    columns_to_extract = ['tweet_id', 'tweet_text', 'class_label']

    # Define disaster types
    disaster_types = ['wildfires', 'cyclone', 'hurricane', 'earthquake', 'floods']

    # Define file types
    file_types = ['train', 'dev', 'test']

    # Define output dataframe
    df_source = pd.DataFrame()


    # Loop through each folder in the folders_path
    for folder_name in os.listdir(folders_path):

        # Define variables for additional columns
        d_year = '' # year of disaster
        disaster = ''  # disaster name

        # Skip the folder ".DS_Store"
        if folder_name.startswith('.'):
            continue

        # Extract year and type of disaster from folder names
        folder_words = folder_name.split(sep='_')
        if folder_words[-1].isdigit():
            if int(folder_words[-1]) > 2000 and int(folder_words[-1]) < 2030:
                d_year = folder_words[-1]
        else:
            continue

        for word in folder_words:
            if word in disaster_types:
                disaster = word

        # Get the path of the folder
        folder_path = os.path.join(folders_path, folder_name)

        # Loop through each TSV file in the folder
        for filename in os.listdir(folder_path):

            usage = '' # file usage

            # Only process TSV files
            if filename.endswith('.tsv'):

                # Get the path of the TSV file
                file_path = os.path.join(folder_path, filename)

                # Load the TSV file into a DataFrame
                df = pd.read_csv(file_path, sep='\t')

                # Extract the desired columns
                df = df[columns_to_extract]

                filename = filename.replace('.tsv', '')
                file_words = filename.split(sep='_')

                # Obtain "train","dev","test"
                for word in file_words:
                    if word in file_types:
                        usage = word

                # Assign additional columns
                df['d_year'] = d_year
                df['disaster'] = disaster
                df['usage'] = usage
            else:
                continue

            # Combine file into one data frame
            df_source = pd.concat([df_source, df])

    return df_source

"""
if __name__ == '__main__':
    extract_tweets_from_files()
else:
    raise Exception("This function can only be called from main.py")
"""


import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer


def clean_text(text, ):

    default_stemmer = PorterStemmer()
    default_stopwords = stopwords.words('english') # or any other list of your choice

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    #text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    #text.strip(' ') # strip whitespaces again?

    return text

# Elbow criterion - Determine optimal numbers of clusters by elbow rule.
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib


def elbow_plot(data, maxK=15, seed_centroids=None):
    """
        parameters:
        - data: pandas DataFrame (data to be fitted)
        - maxK (default = 10): integer (maximum number of clusters with which to run k-means)
        - seed_centroids (default = None ): float (initial value of centroids for k-means)
    """
    sse = []
    K= range(1, maxK)
    for k in K:
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KMeans(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
        else:
            kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            #data["clusters"] = kmeans.labels_
        print("k: ", k,"sse: ",kmeans.inertia_)

        sse.append(kmeans.inertia_)
    # Set the style
    sns.set_style('whitegrid')

    # Create the line chart
    sns.lineplot(x=K, y=sse, color='blue')

    # Set the labels and title
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')

    # Show the plot

    plt.show()
    return kmeans.labels_