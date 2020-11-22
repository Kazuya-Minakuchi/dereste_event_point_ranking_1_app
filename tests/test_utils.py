import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import is_date, is_natural_number, is_positive_number

def test_is_date():
    assert is_date('2020-11-20') == True
    assert is_date('2020-13-20') == False
    assert is_date('2020') == False
    assert is_date('a') == False

def test_is_natural_number():
    assert is_natural_number('10') == True
    assert is_natural_number('1') == True
    assert is_natural_number('0') == False
    assert is_natural_number('0.5') == False
    assert is_natural_number('-5') == False
    assert is_natural_number('a') == False

def test_is_positive_number():
    assert is_positive_number('5.5') == True
    assert is_positive_number('0') == False
    assert is_positive_number('-5.5') == False
    assert is_positive_number('a') == False

