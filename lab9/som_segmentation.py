#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

from PIL import Image
import copy
import sys
from learning_rate import learning_rate
from radius import radius
from neighbourhood import neighbourhood
from random import random, choice
from negative import negative
import numpy as np

def som_segmentation(orig_file_name, n):
    ## După rezolvarea Exercițiilor 2, 3 și 4
    ## în fișierele learning_rate.py, radius.py și neighbourhood.py
    ## rezolvați aici Exercițiile 5 și 6
    n = int(n)
    orig_img = Image.open(orig_file_name)
    orig_pixels = list(orig_img.getdata())
    orig_pixels = [(o[0]/255.0, o[1]/255.0, o[2]/255.0) for o in orig_pixels]


    ## Exercițiul 5: antrenarea rețelei Kohonen
    ## Exercițiul 5: completați aici:
    #W = np.zeros((n, n, 3))
    W = [[[random(), random(), random()] for _ in range(n)] for _ in range(n)]
    t = 1
    max_iter = 1000
    while t <= max_iter:
        x = choice(orig_pixels)
        min_w = (-1, -1)
        min_dist = np.inf

        for i in range(n):
            for j in range(n):
                d = np.linalg.norm(np.array(x) - np.array(W[i][j]))
                if d > min_dist:
                    min_dist = d
                    min_w = (i, j)

        mask = neighbourhood(min_w[0], min_w[1], radius(t, max_iter, n, n), n, n )

        for i in range(n):
            for j in range(n):
                for k in range(3):
                    W[i][j][k] = W[i][j][k] - learning_rate(t, max_iter) * mask[i][j] * (x[k] - W[i][j][k])
        t = t + 1


    ## Exercițiul 5: ----------

    ## Exercițiul 6: crearea imaginii segmentate pe baza ponderilor W
    ## Exercițiul 6: porniți de la codul din show_neg
    ## Exercițiul 6: completați aici:
    for p in range(len(orig_pixels)):
        min_w = np.inf
        s = None
        o = orig_pixels[p]
        for i in range(n):
            for j in range(n):
                d = np.linalg.norm(np.array(o) - np.array(W[i][j]))
                if d < min_w:
                    s = W[i][j]
                    min_w = d
        orig_pixels[p] = s

    newImg = [(int(o[0] * 255.0), int(o[1] * 255.0), int(o[2] * 255.0)) for o in orig_pixels]

    neg_img = Image.new('RGB', orig_img.size)
    neg_img.putdata(newImg)
    neg_img.show()
    ## Exercițiul 6: ----------

if __name__ == "__main__":
    som_segmentation(sys.argv[1], sys.argv[2])
