from numpy.linalg import norm
from numpy import dot


def cosineDistance(a, b):
    return 1 - (dot(a, b) / (norm(a) * norm(b)))


def euclideanDistance(a, b):
    return norm(a-b)
