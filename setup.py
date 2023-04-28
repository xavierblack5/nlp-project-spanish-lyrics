from pysentimiento import create_analyzer
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import nltk
import pickle
from collections import OrderedDict, Counter
from pysentimiento import create_analyzer

# Stanza for sentiment analysis

stopwords = nltk.corpus.stopwords.words('spanish')
stopwords.append(" ")
stopwords.append(",")
stopwords.append(":")
stopwords.append(";")
stopwords.append(".")
stopwords.append("!")
stopwords.append("?")
stopwords.append("'")
stopwords.append("¿")
stopwords.append("\"")
stopwords.append("si")

analyzer = create_analyzer(task="sentiment", lang="es")
analyzer = pickle.load( open( "sent_analyzer.p", "rb" ) )
nlp = pickle.load( open( "pos_tagger.p", "rb" ) )

def parseCSV(filePath):
    # CVS Column Names
    col_names = ['Index','File', 'Name', 'Singer', 'Place', 'Year', 'Genre', 'LyricNumber', 'Composer', 'Sex', 'Lyrics', 'Text', 'Words']
    # Use Pandas to parse the CSV file
    csvData = pd.read_csv(filePath,sep=";",names=col_names, header=0)
    # Get content_words and get a list of each unique word in a song
    csvData["content_words"] = csvData["Words"].apply(lambda x: [item for item in str(x).replace("\"","").replace("'","").replace("[","").replace(" ", "").replace("]","").split(",") if item not in stopwords])
    csvData["unique_words"] = csvData["content_words"].apply(lambda x: np.delete(np.unique(x), np.where(np.unique(x) == '')))
    # Uses spacy for labeling pos for content words
    pos = []
    for text in csvData["Text"]:
        doc = nlp(text)
        pos.append(doc)
    csvData["pos"] = pos 
    csvData["content_pos"] = csvData["pos"].apply(lambda x: [item for item in x if str(item) not in stopwords])
    csvData["content_pos_count"] = csvData["content_pos"].apply(lambda x: OrderedDict(Counter([item.pos_  for item in x]).most_common()))
    csvData["POS"] = csvData["content_pos"].apply(lambda x: [str(item.lemma_ + ": " + item.pos_ + " " + item.dep_)  for item in x])
    
    # Uses pysentimentio for labeling the song as pos, neg, neu and give percentages of those
    sentiment_out = []
    sentiment_neg = []
    sentiment_neu = []
    sentiment_pos = []
    for text in csvData["Text"]:
        res = analyzer.predict(text)
        sentiment_out.append(res.output)
        sentiment_neg.append(res.probas["NEG"])
        sentiment_neu.append(res.probas["NEU"])
        sentiment_pos.append(res.probas["POS"])
    csvData['Sentiment_Output'] = sentiment_out
    csvData['Sentiment_Neg_Probability'] = sentiment_neg
    csvData['Sentiment_Neu_Probability'] = sentiment_neu
    csvData['Sentiment_Pos_Probability'] = sentiment_pos
    return csvData
csv1 = parseCSV("buenos_aires_dataframe.csv")
csv2 = parseCSV("madrid_dataframe.csv")
csv3 = parseCSV("montevideo_dataframe.csv")
csvs = [csv1,csv2,csv3]
all_df = pd.concat(csvs)
trimmed = all_df.drop(['content_pos'], axis=1)
# all_df = all_df.rename({'Unnamed: 0': 'Index'}, axis=1)
pickle.dump( trimmed, open( "lyrics_corpus.p", "wb" ) )


def word_across_song_count(threshold, city):
    wc = Counter()
    if city == 'all':
        cities = all_df['Place'].unique()
        for city in cities:
            city_df = all_df.loc[all_df['Place'] == city]
            uniq_words = city_df["unique_words"].to_numpy()
            uniq_words = np.hstack(uniq_words)
            wc.update(Counter(uniq_words))
        wc = Counter(el for el in wc.elements() if wc[el] >= threshold)
        word_count = sorted(wc.items(), key=lambda item: (-item[1], item[0]))
        all_words = OrderedDict(word_count)
        return all_words
    else:
        city_df = all_df.loc[all_df['Place'] == city]
        uniq_words = city_df["unique_words"].to_numpy()
        uniq_words = np.hstack(uniq_words)
        wc.update(Counter(uniq_words))
        wc = Counter(el for el in wc.elements() if wc[el] >= threshold)
        word_count = sorted(wc.items(), key=lambda item: (-item[1], item[0]))
        all_words = OrderedDict(word_count)
        return all_words

wc_all = word_across_song_count(10, 'all')
wc_ba = word_across_song_count(10, 'BUENOS AIRES')
wc_mad = word_across_song_count(10, 'MADRID')
wc_mv = word_across_song_count(10, 'MONTEVIDEO')
word_counts = [wc_ba, wc_mad, wc_mv, wc_all]
# word_counts = [ wc_ba, wc_all]
pickle.dump( word_counts, open( "word_counts.p", "wb" ) )

def sent_song_count(city):
    if city == 'all':
        songs = all_df['Sentiment_Output'].values.tolist()
        sentcount = Counter(songs)
        sent_count = sorted(sentcount.items(), key=lambda item: (-item[1], item[0]))
        sentiment_count = OrderedDict(sent_count)
        return sentiment_count
    else:
        city_df = all_df.loc[all_df['Place'] == city]
        songs = city_df["Sentiment_Output"].values.tolist()
        sentcount = Counter(songs)
        sent_count = sorted(sentcount.items(), key=lambda item: (-item[1], item[0]))
        sentiment_count = OrderedDict(sent_count)
        return sentiment_count
sc_all = sent_song_count("all")
sc_ba = sent_song_count("BUENOS AIRES")
sc_mad = sent_song_count("MADRID")
sc_mv = sent_song_count("MONTEVIDEO")
sent_counts = [sc_ba, sc_mad, sc_mv, sc_all]
# sent_counts = [sc_ba, sc_all]
pickle.dump( sent_counts, open( "sent_counts.p", "wb" ) )

nlp = spacy.load('es_core_news_md')
pickle.dump( nlp, open( "pos_tagger.p", "wb" ) )

analyzer = create_analyzer(task="sentiment", lang="es")
pickle.dump( analyzer, open( "sent_analyzer.p", "wb" ) )