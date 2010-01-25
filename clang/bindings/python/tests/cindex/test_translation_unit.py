from clang.cindex import *
import os

kInputsDir = os.path.join(os.path.dirname(__file__), 'INPUTS')

def test_spelling():
    path = os.path.join(kInputsDir, 'hello.cpp')
    index = Index.create()
    tu = index.parse(path)
    assert tu.spelling == path

def test_cursor():
    path = os.path.join(kInputsDir, 'hello.cpp')
    index = Index.create()
    tu = index.parse(path)
    c = tu.cursor
    assert isinstance(c, Cursor)
    assert c.kind is CursorKind.TRANSLATION_UNIT

def test_parse_arguments():
    path = os.path.join(kInputsDir, 'parse_arguments.c')
    index = Index.create()
    tu = index.parse(path, ['-DDECL_ONE=hello', '-DDECL_TWO=hi'])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'hello'
    assert spellings[-1] == 'hi'

def test_unsaved_files():
    index = Index.create()
    # FIXME: Why can't we just use "fake.h" here (instead of /tmp/fake.h)?
    tu = index.parse('fake.c', unsaved_files = [
            ('fake.c', """
#include "/tmp/fake.h"
int x;
int SOME_DEFINE;
"""),
            ('/tmp/fake.h', """
#define SOME_DEFINE y
""")
            ])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'x'
    assert spellings[-1] == 'y'
