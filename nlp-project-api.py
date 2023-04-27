from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
import pandas as pd
import pickle
from pysentimiento import create_analyzer
import numpy as np
import nltk
from collections import OrderedDict, Counter
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
  
app = Flask(__name__)

# df = pickle.load( open( "lyrics_corpus.p", "rb" ) )
# trimmed = df.drop(['Text','content_words','unique_words','POS','Lyrics','pos','Words'], axis=1)
# word_counts = pickle.load( open( "word_counts.p", "rb" ) )
# sent_counts = pickle.load( open( "sent_counts.p", "rb" ) )
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

  
  
@app.route('/')
@app.route('/home')
def home():
    df = pickle.load( open( "lyrics_corpus.p", "rb" ) )
    word_counts = pickle.load( open( "word_counts.p", "rb" ) )
    sent_counts = pickle.load( open( "sent_counts.p", "rb" ) )
    places = df['Place'].unique().tolist()
    places.append('All')
    return render_template('home.html', wcs=word_counts, scs=sent_counts, places=places, zip=zip)
  
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

def word_across_song_count(df,threshold, city):
    wc = Counter()
    if city == 'all':
        cities = df['Place'].unique()
        for city in cities:
            city_df = df.loc[df['Place'] == city]
            uniq_words = city_df["unique_words"].to_numpy()
            uniq_words = np.hstack(uniq_words)
            wc.update(Counter(uniq_words))
        wc = Counter(el for el in wc.elements() if wc[el] >= threshold)
        word_count = sorted(wc.items(), key=lambda item: (-item[1], item[0]))
        all_words = OrderedDict(word_count)
        return all_words
    else:
        city_df = df.loc[df['Place'] == city]
        uniq_words = city_df["unique_words"].to_numpy()
        uniq_words = np.hstack(uniq_words)
        wc.update(Counter(uniq_words))
        wc = Counter(el for el in wc.elements() if wc[el] >= threshold)
        word_count = sorted(wc.items(), key=lambda item: (-item[1], item[0]))
        all_words = OrderedDict(word_count)
        return all_words

def sent_song_count(df,city):
    if city == 'all':
        songs = df['Sentiment_Output'].values.tolist()
        sentcount = Counter(songs)
        sent_count = sorted(sentcount.items(), key=lambda item: (-item[1], item[0]))
        sentiment_count = OrderedDict(sent_count)
        return sentiment_count
    else:
        city_df = df.loc[df['Place'] == city]
        songs = city_df["Sentiment_Output"].values.tolist()
        sentcount = Counter(songs)
        sent_count = sorted(sentcount.items(), key=lambda item: (-item[1], item[0]))
        sentiment_count = OrderedDict(sent_count)
        return sentiment_count

@app.route('/join', methods=['GET', 'POST'])
def join():
    df = pickle.load( open( "lyrics_corpus.p", "rb" ) )
    if request.method == 'POST':
        # get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
          # set the file path
            uploaded_file.save(file_path)
          # save the file
            data = parseCSV(file_path)
            data = data.drop(['content_pos'], axis=1)
            new_df = df[df.apply(lambda x: x.values.tolist() not in data.values.tolist(), axis=1)]
            new_df = pd.concat([new_df, data])
            places = new_df['Place'].unique()
            print(places)
            word_counts = []
            sent_counts = []
            for city in places:
                word_counts.append(word_across_song_count(new_df, 10, city))
                sent_counts.append(sent_song_count(new_df, city))
            sent_counts.append(sent_song_count(new_df, 'all'))
            word_counts.append(word_across_song_count(new_df, 10, 'all'))
            pickle.dump( new_df, open( "lyrics_corpus.p", "wb" ) )
            pickle.dump( word_counts, open( "word_counts.p", "wb" ) )
            pickle.dump( sent_counts, open( "sent_counts.p", "wb" ) )
        return redirect(url_for('home'))
    else:
        return render_template('join.html')
  
  
@app.route('/songs')
def songs():
    df = pickle.load( open( "lyrics_corpus.p", "rb" ) )
    trimmed = df.drop(['Text','content_words','unique_words','POS','Lyrics','pos','Words', 'content_pos_count'], axis=1)
    places = df['Place'].unique().tolist()
    return render_template("song_list.html", column_names=trimmed.columns.values, df=trimmed.values.tolist(), places=places,
                            zip=zip)


@app.route('/song')
def song():
    name = request.args.get('name')
    df = pickle.load( open( "lyrics_corpus.p", "rb" ) )
    songrow = df.loc[df['Name'] == name]
    singer = songrow['Singer'].item()
    place = songrow['Place'].item()
    year = songrow['Year'].item()
    genre = songrow['Genre'].item()
    lyricnumber = songrow['LyricNumber'].item()
    lyrics = songrow['Text'].item()
    pos = songrow['POS'].item()
    sent_out = songrow['Sentiment_Output'].item()
    sent_neg = songrow['Sentiment_Neg_Probability'].item()
    sent_neu = songrow['Sentiment_Neu_Probability'].item()
    sent_pos = songrow['Sentiment_Pos_Probability'].item()
    unique_words = songrow['unique_words'].item()
    pos_count = songrow['content_pos_count'].item()
    keys, values = [], []
    for key, value in pos_count.items():
        keys.append(key)
        values.append(value)
        
    return render_template("song.html", name=name, singer=singer, place=place, year=year, genre=genre, lyricnumber=lyricnumber, lyrics=lyrics,
                           pos=list(pos), sent_out=sent_out, sent_neg=sent_neg, sent_neu=sent_neu, sent_pos=sent_pos, unique_words=list(unique_words),
                            keys=list(keys), values=list(values), zip=zip)
  


  
if __name__ == '__main__':
    app.run(debug=False)