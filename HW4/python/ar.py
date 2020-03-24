import numpy as np
import cv2
#Import necessary functions
from loadVid import *
from planarH import *
from matchPics import *

panda_frames = loadVid('../data/ar_source.mov')
panda_frame_num = panda_frames.shape[0]
body = loadVid('../data/book.mov')
body_num, body_H, body_W, _ = body.shape

book_cover = cv2.imread('../data/cv_cover.jpg')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('ar_result.avi', fourcc, 20.0, (body_W, body_H))

for i in range(300):
    print('processing {}th img.....'.format(i))

    f_panda = panda_frames[i % panda_frame_num, :, :, :]
    f_body = body[i, :, :, :]
    f_body = np.squeeze(f_body)

    # adjusting panda frame
    book_H, book_W, _ = book_cover.shape
    panda_H, panda_W, _ = f_panda.shape
    new_W = book_W * (panda_H/book_H)
    panda_cut = f_panda[45:-45, int(panda_W - new_W) // 2 : int(panda_W + new_W)//2]
    panda_new = cv2.resize(panda_cut, dsize=(book_W, book_H))

    matches, loc1, loc2 = matchPics(f_body, book_cover)
    H2to1, _ = computeH_ransac(loc1[matches[:,0]], loc2[matches[:, 1]])
    composite_img = compositeH(H2to1, panda_new, f_body)
    out.write(composite_img)

out.release()
