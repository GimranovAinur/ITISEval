import gensim as gsm

def read_corpus(fname):
    documents = []
    with open(fname) as f:
        for line in f.readlines():
            documents.append(gsm.utils.simple_preprocess(line))
    return documents

documents = read_corpus('habrahabr_corpus.txt')


dict = gsm.corpora.Dictionary(documents)
corpus = [dict.doc2bow(document, allow_update=True) for document in documents]

ldamodel = gsm.models.LdaModel(corpus, num_topics=100, id2word=dict, passes=20)
ldamodel.save('lda.model')

model = gsm.models.LdaModel.load("lda.model")
# print(numpy.exp(-1. * model.log_perplexity(corpus)))

doc1 = read_corpus('data/test_doc1.txt')
doc2 = read_corpus('data/test_doc2.txt')
new_dict1 = gsm.corpora.Dictionary(doc1)
new_corpus1 = [new_dict1.doc2bow(doc, allow_update=True) for doc in doc1]
new_dict2 = gsm.corpora.Dictionary(doc2)
new_corpus2 = [new_dict1.doc2bow(doc, allow_update=True) for doc in doc2]

model.update(new_corpus1)
model.update(new_corpus2)

check_doc1 = list(new_corpus1)[0]
check_doc2 = list(new_corpus2)[0]
print(model.get_document_topics(check_doc1))
print(model.get_document_topics(check_doc2))

