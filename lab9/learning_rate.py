# -*- coding: utf-8 -*-
# Tudor Berariu, 2016

def learning_rate(iter_no, iter_count):
    """Calculează rata de învățare în funcție de iterația curentă și
    numărul total de iterații"""
    ## Exercițiul 2: calculul coeficientului de învățare
    ## Exercițiul 2: completați aici

    MAX_VALUE = 0.75
    MIN_VALUE = 0.1

    lr = MAX_VALUE - (MAX_VALUE - MIN_VALUE) / iter_count *  iter_no # TODO
    ## Exercițiul 2: ----------

    return lr
