from src.utils import select_method, execute_dict_functions, load_pickle

def test_select_method():
    pass

def test_execute_dict_functions():
    input_dict = {
        'a': (lambda: 'b'),
        'c': (lambda: 10),
    }
    assert execute_dict_functions(input_dict) == {'a': 'b', 'c': 10}

def test_load_pickle():
    pass
