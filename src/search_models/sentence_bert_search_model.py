
import numpy as np

from .sentence_search_model import Sentence_SearchModel


class SentenceBERT_SearchModel(Sentence_SearchModel):

    def __init__(self):
        self.model_code = None
        self.model_query = None
        self.dot_model = None


    def generate_embeddings(self, dataset):

        description = [
            dataset.as_numpy_iterator().next()[0][0],
            dataset.as_numpy_iterator().next()[0][1],
            dataset.as_numpy_iterator().next()[0][2],
        ]

        code = [
            dataset.as_numpy_iterator().next()[0][3],
            dataset.as_numpy_iterator().next()[0][4],
            dataset.as_numpy_iterator().next()[0][5],
        ]

        embedded_tokens = self.model_code.predict(code)
        embedded_desc = self.model_query.predict(description)

        return embedded_desc, embedded_tokens

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

            embedded_desc_copy[idx] = (self.model_query.predict([np.array(desc_[0]).reshape((1, -1)),
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