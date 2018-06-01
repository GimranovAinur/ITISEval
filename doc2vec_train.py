import gensim as gsm
import smart_open
import multiprocessing


import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gsm.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gsm.models.doc2vec.TaggedDocument(gsm.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus('data/corpus.txt'))


print(train_corpus)
cpu_count = multiprocessing.cpu_count()

model = gsm.models.doc2vec.Doc2Vec(vector_size=200,
                                   window=10,
                                   min_count=2,
                                   alpha=0.025,
                                   min_alpha=0.025,
                                   workers=cpu_count,
                                   epochs=100)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save('doc2vec.model')
