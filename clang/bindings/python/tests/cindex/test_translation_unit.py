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

def test_reparse_arguments():
    path = os.path.join(kInputsDir, 'parse_arguments.c')
    index = Index.create()
    tu = index.parse(path, ['-DDECL_ONE=hello', '-DDECL_TWO=hi'])
    tu.reparse()
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'hello'
    assert spellings[-1] == 'hi'

def test_unsaved_files():
    index = Index.create()
    tu = index.parse('fake.c', ['-I./'], unsaved_files = [
            ('fake.c', """
#include "fake.h"
int x;
int SOME_DEFINE;
"""),
            ('./fake.h', """
#define SOME_DEFINE y
""")
            ])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-2] == 'x'
    assert spellings[-1] == 'y'

def test_unsaved_files_2():
    import StringIO
    index = Index.create()
    tu = index.parse('fake.c', unsaved_files = [
            ('fake.c', StringIO.StringIO('int x;'))])
    spellings = [c.spelling for c in tu.cursor.get_children()]
    assert spellings[-1] == 'x'

def normpaths_equal(path1, path2):
    """ Compares two paths for equality after normalizing them with
        os.path.normpath
    """
    return os.path.normpath(path1) == os.path.normpath(path2)

def test_includes():
    def eq(expected, actual):
        if not actual.is_input_file:
            return  normpaths_equal(expected[0], actual.source.name) and \
                    normpaths_equal(expected[1], actual.include.name)
        else:
            return normpaths_equal(expected[1], actual.include.name)

    src = os.path.join(kInputsDir, 'include.cpp')
    h1 = os.path.join(kInputsDir, "header1.h")
    h2 = os.path.join(kInputsDir, "header2.h")
    h3 = os.path.join(kInputsDir, "header3.h")
    inc = [(src, h1), (h1, h3), (src, h2), (h2, h3)]

    index = Index.create()
    tu = index.parse(src)
    for i in zip(inc, tu.get_includes()):
        assert eq(i[0], i[1])
