
import os


class CodeSearchManager():

    def __init__(self):
        self.training_model, self.code_model = None, None
        self.desc_model, self.dot_model = None, None

    def load_weights(self, path):
        if os.path.isfile(path + '.index'):
            self.training_model.load_weights(path)
            print("Weights loaded!")
        else:
            print("Warning! No weights loaded!")

    def get_top_n(self, n, results):
        count = 0
        for r in results:
            if results[r] < n:
                count += 1
        return count / len(results)

    def train(self, training_set, weights_path, epochs=1, batch_size=None, steps_per_epoch=None):
        self.training_model.fit(training_set, epochs=epochs, verbose=1, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        self.training_model.save_weights(weights_path)
        print("Model saved!")
