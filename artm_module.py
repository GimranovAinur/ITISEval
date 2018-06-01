
import os
import artm
import glob

os.environ['ARTM_SHARED_LIBRARY'] = '/home/ainur/PycharmProjects/doc2vec_test/bigartm/bigartm/build/lib/libartm.so'

BATCHES_FOLDER = 'data/pydata_batches'
HABR_DATA_PATH = 'habrahabr_corpus_multimodal.txt'
DICT_PATH ='dictionary.dict'
VOCAB_PATH = 'vocabulary.txt'
print("Start")


batch_vec = artm.BatchVectorizer(data_path=HABR_DATA_PATH, data_format='vowpal_wabbit', collection_name='habr', target_folder=BATCHES_FOLDER,vocabulary= VOCAB_PATH, batch_size=100, class_ids={'@word':1})
# batch_vec = artm.BatchVectorizer(data_path=BATCHES_FOLDER, data_format='batches', vocabulary=VOCAB_PATH, gather_dictionary=True)
print("BAtch")
dictionary = artm.Dictionary(data_path=BATCHES_FOLDER)
dictionary.save_text(dictionary_path=DICT_PATH)
model = artm.ARTM(num_topics=10,
                  num_document_passes=10,#10 проходов по документу
                  dictionary=dictionary,
                  scores=[artm.TopTokensScore(name='top_tokens_score')])
model.fit_offline(batch_vectorizer=batch_vec, num_collection_passes=10)
top_tokens = model.score_tracker['top_tokens_score']

# print('Sparsity Phi:{1:.3f} (ARTM)'.format(
#         model_artm.score_tracker['SparsityPhiScore'].last_value))