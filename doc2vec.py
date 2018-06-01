import gensim as gsm
import smart_open
from scipy.spatial.distance import cosine


train_corpus = list(read_corpus('habrahabr_corpus.txt'))
model = gsm.models.Doc2Vec.load('doc2vec.model')

test_paper1 = 'data/test_doc1.txt'
test_paper2 = 'data/test_doc2.txt'
new_test1 = list(read_corpus(test_paper1, tokens_only=True))
new_test2 = list(read_corpus(test_paper2, tokens_only=True))

inferred_docvec1 = model.infer_vector(new_test1.pop())
inferred_docvec2 = model.infer_vector(new_test2.pop())

print(model.docvecs.most_similar([inferred_docvec2], topn=1))
print(model.docvecs.most_similar([inferred_docvec1], topn=1))


print(cosine(inferred_docvec1, inferred_docvec2))

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gsm.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gsm.models.doc2vec.TaggedDocument(gsm.utils.simple_preprocess(line), [i])
