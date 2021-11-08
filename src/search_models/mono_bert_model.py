
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
from .. import helper

from .code_search_manager import CodeSearchManager

class MonoBERT_SearchModel(CodeSearchManager):

    def __init__(self):
        pass

    def test_embedded(self, dataset, results_path, number_of_elements=100 ):

        np_dataset = dataset.as_numpy_iterator().next()

        test_desc = np_dataset[0]
        test_code_str = np_dataset[1]

        #test_tokens = helper.load_hdf5(self.data_path + "test.tokens.h5", 0, number_of_elements)
        #test_desc = helper.load_hdf5(self.data_path + "test.desc.h5" , 0, number_of_elements)

        #for token in test_tokens:
        #    code = token #(" ".join([self.vocab_tokens[x] for x in token]))
        #    test_code_str.append(code)


        results = {}
        pbar = tqdm(total=len(test_desc))

        for rowid, desc in enumerate(test_desc):

            desc = test_desc[rowid] #(" ".join([self.vocab_desc[x] for x in test_desc[rowid]]))
            results[rowid] = self.get_id_rank(rowid, desc, test_code_str)
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

    def get_id_rank(self, rowid, desc_string, test_tokens_str):
        desc = desc_string

        # Create array same length as the rest of code embeddings only containing this description
        tiled_desc = [desc] * len(test_tokens_str)

        input_ids, attention_mask, token_type_ids = self.tokenize_sentences(tiled_desc, test_tokens_str)

        candidate_prediction = self.training_model.predict([input_ids,
                                                attention_mask,
                                                token_type_ids])

        candidate_prediction = candidate_prediction.reshape((-1))

        # get ids sorted by prediction value
        candidate_prediction = candidate_prediction.argsort()[::-1]

        return np.where(candidate_prediction == rowid)[0][0]

    def test(self, dataset, results_path, number_of_elements=100 ):

        #self.test_embedded(dataset, results_path, number_of_elements )

        df = pd.read_csv(self.data_path + "descriptions.csv", header=0)
        df = df.dropna()
        df = df[df["rowid"] < number_of_elements]

        self.rephrasing_test(df, number_of_elements)


    def rephrasing_test(self, rephrased_descriptions_df,  number_of_elements=100 ):

        test_tokens = helper.load_hdf5(self.data_path + "test.tokens.h5", 0, number_of_elements)
        test_code_str = []
        for token in test_tokens:
            code = (" ".join([self.vocab_tokens[x] for x in token]))
            test_code_str.append(code)

        rephrased_ranking = {}
        new_ranking = {}
        pbar = tqdm(total=len(rephrased_descriptions_df.index))
        for i, row in enumerate(rephrased_descriptions_df.iterrows()):
            idx = row[1].values[0]
            original_desc = row[1].values[1]
            new_desc = row[1].values[2]

            rephrased_ranking[idx] = self.get_id_rank(idx, original_desc, test_code_str)
            new_ranking[idx] = self.get_id_rank(idx, new_desc, test_code_str)
            pbar.update(1)
        pbar.close()


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