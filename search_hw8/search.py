import pandas as pd
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def build_index():
    df = pd.read_csv('News.csv')
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'id', 'publication', 'author', 'date', 'year', 'month', 'url'], axis=1)
    df['new_content'] = df['content'].apply(remove_stop_words)
    return df

def remove_stop_words(sir):
    sir = sir.lower()
    stopword = set(stopwords.words('english'))
    splited = sir.split(' ')
    splited = [word for word in splited if word not in stopword]
    return ' '.join(splited)

def retrieve(query, df):
    candidates, summa = [], 0
    content = [con for con in df['new_content']]
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    tfidf_wm = tfidfvectorizer.fit_transform(content)
    tfidf_tokens = tfidfvectorizer.get_feature_names()
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
    query = remove_stop_words(query).split()
    for i in range(len(df_tfidfvect)):
        for quer in query:
            try:
                summa += df_tfidfvect[quer][i]
            except:
                summa += 0.0
        candidates.append((summa, i))
        summa = 0
    candidates = sorted(candidates, reverse=True)
    return candidates[:10]
