# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

def radius(iter_no, iter_count, height, width):
    """Calculează raza în funcție de iterația curentă,
    numărul total de iterații și dimensiunea rețelei"""
    ## Exercițiul 3: calculul razei în funcție de dimensiunile rețelei
    ##           și de iterația curentă Exercițiul 3: completați aici

    ## Exercițiul 3: ----------
    b =  max(width, height) / 2
    if iter_no % 50 == 0:
    	print(b - 0.7 * b * iter_no / iter_count)
    return b - 0.7 * b * iter_no / iter_count