
import os
from tqdm import tqdm
import numpy as np
import time
import random
from . import help

class CodeSearchManager():

    def __init__(self):
        self.training_model, self.code_model = None, None
        self.desc_model, self.dot_model = None, None
        self.chunk_size = 600000

    def get_dataset_meta(self):
        raise NotImplementedError(self)

    def get_dataset_meta_hardcoded(self):
        raise NotImplementedError()

    def generate_model(self):
        raise NotImplementedError()

    def tokenize(self, sentence):
        raise NotImplementedError()

    def load_weights(self, path):
        if os.path.isfile(path + '.index'):
            self.training_model.load_weights(path)
            print("Weights loaded!")
        else:
            print("Warning! No weights loaded!")

    def training_data_chunk(self, id, valid_perc=1.0):

        init_trainig = self.chunk_size * id
        init_valid = int(self.chunk_size * id + self.chunk_size * valid_perc)
        end_valid = int(self.chunk_size * id + self.chunk_size)

        return init_trainig, init_valid, end_valid

    def get_top_n(self, n, results):
        count = 0
        for r in results:
            if results[r] < n:
                count += 1
        return count / len(results)

    def train(self, training_set, weights_path, epochs=1, batch_size=None):
        self.training_model.fit(training_set, epochs=epochs, verbose=1, batch_size=batch_size)
        self.training_model.save_weights(weights_path)
        print("Model saved!")


    def get_id_rank(self, rowid, embedded_tokens, embedded_desc):

        # Get the similarity with the ground truth
        expected_best_result = \
        self.dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]

        # Now we compare this desc with the rest of the code snippets
        # Remove this description row from the list of code embeddings
        deleted_tokens = np.delete(embedded_tokens, rowid, 0)

        # Create array same length as the rest of code embeddings only containing this description
        tiled_desc = np.tile(embedded_desc[rowid], (deleted_tokens.shape[0], 1))

        # Similarity between this description and the rest of code embeddings
        prediction = self.dot_model.predict([deleted_tokens, tiled_desc], batch_size=32 * 4)

        return len(prediction[prediction >= expected_best_result])


    def test_embedded(self, embedded_tokens, embedded_desc, results_path):

        results = {}
        pbar = tqdm(total=len(embedded_desc))

        for rowid, desc in enumerate(embedded_desc):

            results[rowid] = self.get_id_rank(rowid, embedded_tokens, embedded_desc)

            pbar.update(1)
        pbar.close()

        top_1 = self.get_top_n(1, results)
        top_3 = self.get_top_n(3, results)
        top_5 = self.get_top_n(5, results)
        top_15 = self.get_top_n(15, results)

        print(top_1)
        print(top_3)
        print(top_5)
        print(top_15)

        name = results_path + time.strftime("%Y%m%d-%H%M%S") + ".csv"

        f = open(name, "a")

        f.write("top1,top3,top5\n")
        f.write( str(top_1) + "," + str(top_3) + "," + str(top_5) + "\n")
        f.close()

        help.save_pickle(results_path + time.strftime("%Y%m%d-%H%M%S")+ "-rankings" + ".pkl", results)

    def rephrasing_test(self, rephrased_descriptions_df, embedded_tokens, embedded_desc):

        rephrased_ranking = {}
        new_ranking = {}
        for i, row in enumerate(rephrased_descriptions_df.iterrows()):
            idx = row[1].values[0]

            original_desc = row[1].values[1]

            embedded_tokens_copy = embedded_tokens.copy()
            embedded_desc_copy = embedded_desc.copy()

            original_rank = self.get_id_rank(idx, embedded_tokens_copy, embedded_desc_copy)

            desc = row[1].values[2]

            desc_ = self.tokenize(desc)

            embedded_desc_copy[idx] = (self.desc_model.predict([np.array(desc_[0]).reshape((1, -1)),
                                                            np.array(desc_[1]).reshape((1, -1)),
                                                            np.array(desc_[2]).reshape((1, -1))

                                                            ])[0])

            new_rank = self.get_id_rank(idx, embedded_tokens_copy, embedded_desc_copy)

            rephrased_ranking[idx] = original_rank
            new_ranking[idx] = new_rank

        print("Number of queries: ",str(len(rephrased_descriptions_df.index)))
        print("Selected topN:")
        print(self.get_top_n(1, rephrased_ranking))
        print(self.get_top_n(3, rephrased_ranking))
        print(self.get_top_n(5, rephrased_ranking))

        print("Rephrased topN:")
        print(self.get_top_n(1, new_ranking))
        print(self.get_top_n(3, new_ranking))
        print(self.get_top_n(5, new_ranking))
        return rephrased_ranking, new_ranking

    def generate_similarity_examples(self, embedded_tokens, embedded_desc, dot_model, results_path):
        f = open(results_path + time.strftime("%Y%m%d-%H%M%S")+ "-similarities" + ".csv", "a")
        f.write("match,similarity\n")

        for i in range(0, len(embedded_tokens)):
            similarity = dot_model.predict([embedded_tokens[i].reshape((1, -1)), embedded_desc[i].reshape((1, -1))])[0][0]
            f.write( str(1) + "," + str(similarity) + "\n")


        for i in range(0, len(embedded_tokens)):
            random_code = random.randint(0, len(embedded_tokens)-1)
            random_desc = random.randint(0, len(embedded_tokens)-1)

            similarity = dot_model.predict([embedded_tokens[random_code].reshape((1, -1)), embedded_desc[random_desc].reshape((1, -1))])[0][0]
            f.write( str(0) + "," + str(similarity) + "\n")

        f.close()