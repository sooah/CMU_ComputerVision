import multiprocessing
import pickle
import math
import cv2 as cv
from python.createFilterBank import create_filterbank
from python.getVisualWords import get_visual_words


def worker_to_visual_words(wind):
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    all_imagenames = meta['all_imagenames']
    point_method = 'Random'
    dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))
    filterBank = create_filterbank()

    for j in range(math.ceil(len(all_imagenames) / num_cores)):
        img_ind = j * num_cores + wind
        if img_ind < len(all_imagenames):
            img_name = all_imagenames[img_ind]
            print('converting %d-th image %s to visual words' % (img_ind, img_name))
            image = cv.imread('../data/%s' % img_name)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert the image from bgr to rgb
            wordMap = get_visual_words(image, dictionary, filterBank)
            pickle.dump(wordMap, open('../data/%s_%s.pkl' % (img_name[:-4], point_method), 'wb'))


if __name__ == '__main__':

    num_cores = 2
    workers = []
    for i in range(num_cores):
        workers.append(multiprocessing.Process(target=worker_to_visual_words, args=(i,)))
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    print('batch to visual words done!')