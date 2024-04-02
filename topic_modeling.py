import gensim
from gensim import corpora

def get_topic_clusters(conversation_history):
    if not conversation_history:
        return []  # Return an empty list if conversation_history is empty

    texts = [conv['response'].split() for conv in conversation_history]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    if not dictionary.token2id:
        return []  # Return an empty list if there are no valid terms in the dictionary

    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=5)

    topics = lda_model.print_topics(num_words=4)
    topic_clusters = [[] for _ in range(len(topics))]

    for i, conv in enumerate(conversation_history):
        bow_vector = dictionary.doc2bow(conv['response'].split())
        topic_probabilities = lda_model.get_document_topics(bow_vector)
        top_topic_index = max(topic_probabilities, key=lambda x: x[1])[0]
        topic_clusters[top_topic_index].append(conv)

    return topic_clusters