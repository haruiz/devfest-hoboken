import pickle
import tensorflow as tf


class ModelLoader:
    def __init__(self, path: str, name: str, version: int = 1.0, backend="sklearn"):
        self.backend = backend
        self.name = name
        self.version = version
        self.path = path
        if self.backend == "sklearn":
            self.model = self.__load_model_from_sklearn(self.path)
        elif self.backend == "torch":
            self.model = self.__load_model_from_torch(self.path)
        elif self.backend == "tensorflow":
            self.model = self.__load_model_from_tensorflow(self.path)
        else:
            raise NotImplementedError

    def __load_model_from_sklearn(self, model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def __load_model_from_torch(self, model_path):
        raise NotImplementedError

    def __load_model_from_tensorflow(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    def predict(self, data):
        return self.model.predict(data)

    def __call__(self, data):
        return self.predict(data)  # this is the same as the predict method
