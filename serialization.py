import pickle

def save(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj
