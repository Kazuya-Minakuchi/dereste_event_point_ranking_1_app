from test.support import captured_stdin

import sys, os
path = os.path.join(os.path.dirname(__file__), '../src')
sys.path.append(path)

from selection import Selections
from selection import Selection

"""
def test_selections():
    selections = Selections([
        Selection(
            name = 'test1',
            method = (lambda: 'tes1'),
        ),
        Selection(
            name = 'test2',
            method = (lambda: 'tes2'),
        ),
    ])
    
    with captured_stdin() as stdin:
        stdin.write('0')
        stdin.seek(0)
        assert selections.select_method() == 'tes1'
    with captured_stdin() as stdin:
        stdin.write('1')
        stdin.seek(0)
        assert selections.select_method() == 'tes2'
"""

def test_selection():
    selection = Selection(
        name = 'test', 
        method = (lambda: 'tes'),
    )
    assert selection.name == 'test'
    assert selection.method() == 'tes'
