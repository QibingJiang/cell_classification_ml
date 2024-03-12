
import numpy as np
print("hello")
"""Python module demonstrates passing MATLAB types to Python functions"""
import numpy as np
def search(words):
    """Return list of words containing 'son'"""
    newlist = [w for w in words if 'son' in w]
    a = np.zeros(5)
    print("hello", a)
    return newlist

def theend(words):
    """Append 'The End' to list of words"""
    words.append('The End')
    return words
