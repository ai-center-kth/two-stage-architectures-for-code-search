
from tqdm import tqdm
import numpy as np
import time
import random
import pandas as pd
from .. import helper
from .code_search_manager import CodeSearchManager


class Sentence_SearchModel(CodeSearchManager):

    def __init__(self):
        self.model_code = None
        self.model_query = None
        self.dot_model = None

    def desc_tokenizer(self, desc):
        raise NotImplementedError(self)

    def code_tokenizer(self, code):
        raise NotImplementedError(self)

    def test(self, test_ds, results_path):

        embedded_desc, embedded_tokens = self.generate_embeddings(test_ds)
        self.test_embedded(embedded_tokens, embedded_desc, results_path)

        df = pd.read_csv(self.data_path + "descriptions.csv", header=0)
        df = df.dropna()
        df = df[df["rowid"] < 100]

        self.rephrasing_test(df, embedded_tokens, embedded_desc)


    def generate_embeddings(self, dataset):

        description = dataset.as_numpy_iterator().next()[0][0]

        code = dataset.as_numpy_iterator().next()[0][1]

        print("Embedding tokens...")
        embedded_tokens = self.model_code.predict(code)
        embedded_desc = self.model_query.predict(description)

        return embedded_desc, embedded_tokens


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

        helper.save_pickle(results_path + time.strftime("%Y%m%d-%H%M%S")+ "-rankings" + ".pkl", results)


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

            desc_ = self.desc_tokenizer(desc)

            embedded_desc_copy[idx] = self.model_query.predict(np.array(desc_).reshape(1, -1))[0]


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