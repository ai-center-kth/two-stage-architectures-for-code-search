
import os
from tqdm import tqdm
import numpy as np
import time
import help
import random

class CodeSearchManager():

    def get_dataset_meta(self):
        raise NotImplementedError(self)

    def get_dataset_meta_hardcoded(self):
        raise NotImplementedError()

    def generate_model(self):
        raise NotImplementedError()

    def load_weights(self, model, path):
        if os.path.isfile(path + '.index'):
            model.load_weights(path)
            print("Weights loaded!")
        else:
            print("Warning! No weights loaded!")

    def get_top_n(self, n, results):
        count = 0
        for r in results:
            if results[r] < n:
                count += 1
        return count / len(results)

    def train(self, trainig_model, training_set_generator, weights_path, epochs=1):
        trainig_model.fit(training_set_generator, epochs=epochs)
        trainig_model.save_weights(weights_path)
        print("Model saved!")

    def test_embedded(self, dot_model, embedded_tokens, embedded_desc, results_path):

        results = {}
        pbar = tqdm(total=len(embedded_desc))

        for rowid, desc in enumerate(embedded_desc):

            # Get the similarity with the ground truth
            expected_best_result = dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]

            # Now we compare this desc with the rest of the code snippets
            # Remove this description row from the list of code embeddings
            deleted_tokens = np.delete(embedded_tokens, rowid, 0)

            # Create array same length as the rest of code embeddings only containing this description
            tiled_desc = np.tile(desc, (deleted_tokens.shape[0], 1))

            # Similarity between this description and the rest of code embeddings
            prediction = dot_model.predict([deleted_tokens, tiled_desc], batch_size=32 * 4)

            # Store the number of predictions with better performance
            results[rowid] = len(prediction[prediction > expected_best_result])

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