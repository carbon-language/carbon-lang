from clang.cindex import *
import os

kInputsDir = os.path.join(os.path.dirname(__file__), 'INPUTS')

def test_spelling():
    path = os.path.join(kInputsDir, 'hello.cpp')
    index = Index.create()
    tu = index.parse(path)
    assert str(tu.spelling) == path
