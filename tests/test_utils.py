import sys, os
path = os.path.join(os.path.dirname(__file__), '../src')
sys.path.append(path)

from app.utils import execute_dict_functions, load_pickle

def test_execute_dict_functions():
    input_dict = {
        'a': (lambda: 'b'),
        'c': (lambda: 10),
    }
    assert execute_dict_functions(input_dict) == {'a': 'b', 'c': 10}

def test_load_pickle():
    pass
