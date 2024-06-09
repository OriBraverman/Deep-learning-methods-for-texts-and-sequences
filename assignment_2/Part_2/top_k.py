from assignment_2 import utils


if __name__ == '__main__':

    # Load the pre-trained word2vec using the utils
    word2vec = utils.load_word2vec()

    List_of_words = ["dog", "england", "john", "explode", "office"]

    # For each word in the list, print the k most similar words to it according to the cosine distance.

    for word in List_of_words:
        print("{0} :".format(word), end=" ")
        for i, i_similar in enumerate(utils.most_similar(word, 5, word2vec), 1):
            print("{1} {2}".format(word, i_similar[0], i_similar[1]), end="\n" if i == 5 else ", ")