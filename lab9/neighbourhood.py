#!/usr/bin/python
# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

import sys
import numpy

def neighbourhood(y, x, radius, height, width):
    """Construiește o mască cu valori în intevalul [0, 1] pentru
    actualizarea ponderilor"""
    ## Exercițiul 4: calculul vecinătății
    ## Exercițiul 4: completați aici
    x -= 1
    y -= 1
    mask = [[0 for j in range(width)] for i in range(height)]

    for i in range(width):
        for j in range(height):
            a = numpy.array([x, y])
            b = numpy.array([i, j])
            d = numpy.linalg.norm(a-b)
            if d < radius:
                mask[i][j] = (radius - d) / radius
    ## Exercițiul 4: ----------

    return mask

if __name__ == "__main__":
    m = neighbourhood(int(sys.argv[1]), int(sys.argv[2]),                 # y, x
                      float(sys.argv[3]),                               # radius
                      int(sys.argv[4]), int(sys.argv[5]))        # height, width
    print(m)
