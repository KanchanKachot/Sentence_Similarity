from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def sent_similarity(sentences):
    sentence_embeddings = model.encode(sentences)
    for sentence, embedding in zip(sentences, sentence_embeddings):
        "Sentence:", sentence
        "Embedding:", embedding
    sent_simi='Similarity between {} and {} is {}'.format(sentences[0],sentences[1],cosine_similarity(sentence_embeddings[0].reshape(1, -1),sentence_embeddings[1].reshape(1, -1))[0][0])

    return sent_simi

if __name__=="__main__":
    sentences = ["This is a sentence","What is the Temperature outside?"]
    output=sent_similarity(sentences)
    print(output)