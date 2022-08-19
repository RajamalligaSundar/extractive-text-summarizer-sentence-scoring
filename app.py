from flask import Flask, render_template, request
import os
import urllib
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import re
import PyPDF2 as pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np
import docx2txt as dt
from rouge import Rouge

app = Flask(__name__, template_folder='template')

app.secret_key = "12345"
app.config['UPLOAD_FOLDER1'] = "static/files/pdf"
app.config['UPLOAD_FOLDER2'] = "static/files/text"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/url_summarization")
def url_summarization():
    return render_template("url_summarization.html")


@app.route("/url_fetch", methods=["post"])
def url_fetch():
    if request.method == "POST":
        url = request.form.get('url')

        # scrapping data from url
        source = urllib.request.urlopen(url).read()

        # parse data
        soup = bs(source, 'lxml')

        # pre-processing
        # getting all textdata with paragraph tag
        # raw-document = soup.find_all('p')

        document = soup.get_text()

        nlp = English()
        nlp.add_pipe('sentencizer')

        doc = nlp(document.replace("\n", ""))
        sentences = [sent.text.strip() for sent in doc.sents]

        # Scored sentences in their correct order
        sentence_organizer = {k: v for v, k in enumerate(sentences)}

        # Creating a tf-idf (Term frequnecy Inverse Document Frequency) model
        tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            ngram_range=(1, 3),
                                            use_idf=1, smooth_idf=1,
                                            sublinear_tf=1,
                                            stop_words='english')

        # Passing sentences - treating each as one document to TF-IDF vectorizer
        tf_idf_vectorizer.fit(sentences)

        # Transforming sentences to TF-IDF vectors
        sentence_vectors = tf_idf_vectorizer.transform(sentences)

        # Getting sentence scores for each sentences
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

        # Getting top-n sentences
        N = 3

        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]

        # Mapping the scored sentences with their indexes
        mapped_top_n_sentences = [(sentence, sentence_organizer[sentence]) for sentence in top_n_sentences]

        # Ordering our top-n sentences in their original ordering
        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key=lambda x: x[1])
        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]

        # Final summary
        summary = " ".join(ordered_scored_sentences)

        transformed_summary = re.sub(r'\[[0-9]*]', ' ', summary)

        rouge = Rouge()
        score = rouge.get_scores(summary,document)

    return render_template("url_fetch.html", url=url, summary=summary, transformed_summary=transformed_summary, score=score)


@app.route("/corpus_summarization")
def corpus_summarization():
    return render_template("corpus_summarization.html")


@app.route("/corpus_fetch", methods=["post"])
def corpus_fetch():
    if request.method == "POST":
        corpus = request.form.get('corpus')

        nlp = English()
        nlp.add_pipe('sentencizer')

        doc = nlp(corpus.replace("\n", ""))
        sentences = [sent.text.strip() for sent in doc.sents]

        # scored sentences in their correct order
        sentence_organizer = {k: v for v, k in enumerate(sentences)}
        # Creating a tf-idf (Term frequnecy Inverse Document Frequency) model
        tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            ngram_range=(1, 3),
                                            use_idf=1, smooth_idf=1,
                                            sublinear_tf=1,
                                            stop_words='english')

        # Passing our sentences treating each as one document to TF-IDF vectorizer
        tf_idf_vectorizer.fit(sentences)
        # Transforming our sentences to TF-IDF vectors
        sentence_vectors = tf_idf_vectorizer.transform(sentences)
        # Getting sentence scores for each sentences
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
        # Getting top-n sentences
        N = 3
        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]

        # Let's map the scored sentences with their indexes
        mapped_top_n_sentences = [(sentence, sentence_organizer[sentence]) for sentence in top_n_sentences]
        # Ordering our top-n sentences in their original ordering
        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key=lambda x: x[1])
        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
        # Summary
        summary = " ".join(ordered_scored_sentences)

        transformed_summary = re.sub(r'\[[0-9]*]', ' ', summary)

        rouge = Rouge()
        score = rouge.get_scores(summary,corpus)

    return render_template("corpus_fetch.html", sentences=sentences, transformed_summary=transformed_summary, score=score)


@app.route("/pdf_summarization")
def pdf_summarization():
    return render_template("pdf_summarization.html")


@app.route("/pdf_fetch", methods=["GET", "POST"])
def pdf_fetch():
    if request.method == 'POST':
        upload = request.files['upload']

        if upload.filename != '':
            filepath1 = os.path.join(app.config["UPLOAD_FOLDER1"], upload.filename)

        upload.save(filepath1)

        file = pdf.PdfFileReader(upload)
        #pages = pdf.getNumpages()

        page = file.getPage(0)
        page_content = page.extractText()

        nlp = English()
        nlp.add_pipe('sentencizer')

        doc = nlp(page_content.replace("\n", ""))
        sentences = [sent.text.strip() for sent in doc.sents]

        # scored sentences in their correct order
        sentence_organizer = {k: v for v, k in enumerate(sentences)}
        # Let's now create a tf-idf (Term frequnecy Inverse Document Frequency) model
        tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            ngram_range=(1, 3),
                                            use_idf=1, smooth_idf=1,
                                            sublinear_tf=1,
                                            stop_words='english')
        # Passing our sentences treating each as one document to TF-IDF vectorizer
        tf_idf_vectorizer.fit(sentences)
        # Transforming our sentences to TF-IDF vectors
        sentence_vectors = tf_idf_vectorizer.transform(sentences)
        # Getting sentence scores for each sentences
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
        # Getting top-n sentences
        N = 5
        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]

        # Mapping the scored sentences with their indexes
        mapped_top_n_sentences = [(sentence, sentence_organizer[sentence]) for sentence in top_n_sentences]
        # Ordering our top-n sentences in their original ordering
        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key=lambda x: x[1])
        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
        # Final summary
        summary = " ".join(ordered_scored_sentences)

        rouge = Rouge()
        score = rouge.get_scores(summary, page_content)

    return render_template("pdf_fetch.html", file=file, page_content=page_content, summary=summary, score=score)


@app.route("/text_summarization")
def text_summarization():
    return render_template("text_summarization.html")


@app.route("/text_fetch", methods=["GET", "POST"])
def text_fetch():
    if request.method == 'POST':
        upload_text = request.files['upload_text']

        if upload_text.filename != '':
            filepath2 = os.path.join(app.config["UPLOAD_FOLDER2"], upload_text.filename)

        upload_text.save(filepath2)

        text = dt.process(upload_text)

        nlp = English()
        nlp.add_pipe('sentencizer')

        doc = nlp(text.replace("\n", ""))
        sentences = [sent.text.strip() for sent in doc.sents]

        # scored sentences in their correct order
        sentence_organizer = {k: v for v, k in enumerate(sentences)}
        # Creating a tf-idf (Term frequnecy Inverse Document Frequency) model
        tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                            strip_accents='unicode',
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            ngram_range=(1, 3),
                                            use_idf=1, smooth_idf=1,
                                            sublinear_tf=1,
                                            stop_words='english')
        # Passing our sentences treating each as one document to TF-IDF vectorizer
        tf_idf_vectorizer.fit(sentences)
        # Transforming our sentences to TF-IDF vectors
        sentence_vectors = tf_idf_vectorizer.transform(sentences)
        # Getting sentence scores for each sentences
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
        # Getting top-n sentences
        N = 5
        top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]

        # mapping the scored sentences with their indexes
        mapped_top_n_sentences = [(sentence, sentence_organizer[sentence]) for sentence in top_n_sentences]
        # Ordering our top-n sentences in their original ordering
        mapped_top_n_sentences = sorted(mapped_top_n_sentences, key=lambda x: x[1])
        ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]
        # Final summary
        summary = " ".join(ordered_scored_sentences)

        rouge = Rouge()
        score = rouge.get_scores(summary, text)

    return render_template("text_fetch.html", text=text, summary=summary, score=score)


'''
Checking whether the current file is a main file or not
debug = true - to autorun when changes are made
'''
if __name__ == '__main__':
    app.run(debug=True)
