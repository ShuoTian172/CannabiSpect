from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

stopwords = {line.rstrip().lower(): None for line in open('stopwords.txt')}

def handle_text(text, isUseStopWords):
    if isUseStopWords:
        # new_word_list = [word for word in word_tokenize(text) if word not in stopwords.words('english')]
        new_word_list = [word for word in word_tokenize(text) if word.lower() not in stopwords]
    else:
        new_word_list = [word for word in word_tokenize(text)]

    return new_word_list
