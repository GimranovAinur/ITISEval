
import collections
from bs4 import BeautifulSoup


import nltk
from nltk.corpus import stopwords
import html2text
html2text.config.INLINE_LINKS = False
html2text.config.SKIP_INTERNAL_LINKS = True
html2text.config.IGNORE_ANCHORS = True
html2text.config.IGNORE_EMPHASIS = True
html2text.config.IGNORE_IMAGES = True
html2text.config.BYPASS_TABLES = False

import os
import json

DATASET_PATH = 'data/Habrahabr/habr_posts_20150413_10k/habr_posts'
post_ids = sorted(int(filename) for filename in os.listdir(DATASET_PATH) if not filename.startswith('.'))
def load_post(post_id):
    with open(os.path.join(DATASET_PATH, str(post_id))) as fin:
        post = json.load(fin)
    return post

# post_index = 101121
# post = load_post(101121)
#
#
# content_text = html2text.html2text(post['content_html'])
# tokens = nltk.tokenize.wordpunct_tokenize(content_text)

import pymorphy2
morph = pymorphy2.MorphAnalyzer()


def html_to_plain(html):
    text = html2text.html2text(post['content_html'])


def post_to_corpus_line(post_id, post, morph):

    # 1. words
    words = collections.Counter()
    soup = BeautifulSoup(post['content_html'], "lxml")
    text_parts = soup.findAll(text=True)
    content_text = ''.join(text_parts)

    import nltk
    from nltk.corpus import stopwords
    import pymorphy2

    morph = pymorphy2.MorphAnalyzer()

    words = collections.Counter()
    space_chars = '«»“”’*…/_.\\'
    stop_words = set(stopwords.words('russian'))
    for c in space_chars:
        content_text = content_text.replace(c, ' ')
    tokens = nltk.tokenize.wordpunct_tokenize(content_text)
    tokens = nltk.word_tokenize(content_text)
    tokens = [w for w in tokens if not w in stop_words]
    for token in tokens:
        if len(token) > 2:
            token = token.lower().replace('ё', 'е')
            word = morph.parse(token)[0].normal_form
            if len(word) > 0:
                words[word] += 1

    # 2. users
    users = collections.Counter()

    def pass_comments(comments, users):
        for comment in comments:
            if not comment['banned'] and comment.get('username') is not None:
                username = comment['username']
                users[username] += 1
            pass_comments(comment['replies'], users)

    pass_comments(post['comments'], users)



    # construct bag-of-words

    def construct_bow(words):
        return [
            (
                    word.replace(' ', '_').replace(':', '_').replace('|', '_').replace('\t', '_') +
                    ('' if cnt == 1 else ':%g' % cnt)
            )
            for word, cnt in words.items()
        ]

    parts = (
            ['\'%d' % post_id] +
            ['|@words'] + construct_bow(words)
    )
    return ' '.join(parts)

with open('habrahabr_corpus_multimodal.txt', 'w') as fout:
    for post_id in post_ids:
        line = post_to_corpus_line(post_id, load_post(post_id), morph)
        fout.write(line + '\n')
