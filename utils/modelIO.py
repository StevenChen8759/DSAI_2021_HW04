import pickle

def load(filename: str):
    return pickle.load(open(filename, "rb"))

def save(model, filename: str):
    pickle.dump(model, open(f"./output/{filename}.model", "wb"))
