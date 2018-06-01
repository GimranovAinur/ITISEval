import pymorphy2
import nltk
from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()

words = []
content_text = 'Txt file path'
space_chars = '«»“”’*…/_.\\'
for c in space_chars:
    content_text = content_text.replace(c, ' ')
    stop_words = set(stopwords.words('russian'))
    tokens = nltk.tokenize.wordpunct_tokenize(content_text)
    tokens = nltk.word_tokenize(content_text)
    tokens = [w for w in tokens if not w in stop_words]
    for token in tokens:
        if len(token) > 2:
            token = token.lower().replace('ё', 'е')
            word = morph.parse(token)[0].normal_form
            if len(word) > 0:
                words.append(word)


with open('data/doc.txt', 'w') as fout:
    fout.write(' '.join(words))