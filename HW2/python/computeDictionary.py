import pickle

def compute_dict(dictname, dict):
    with open(dictname, 'wb') as handle:
        pickle.dump(dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return 0
