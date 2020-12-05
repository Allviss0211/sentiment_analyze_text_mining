
import tensorflow_datasets as tfds

def loadData():
    return tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)