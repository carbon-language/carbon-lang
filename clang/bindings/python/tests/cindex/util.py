# This file provides common utility functions for the test suite.

from clang.cindex import Cursor
from clang.cindex import Index

def get_tu(source, lang='c', all_warnings=False):
    """Obtain a translation unit from source and language.

    By default, the translation unit is created from source file "t.<ext>"
    where <ext> is the default file extension for the specified language. By
    default it is C, so "t.c" is the default file name.

    Supported languages are {c, cpp, objc}.

    all_warnings is a convenience argument to enable all compiler warnings.
    """
    name = 't.c'
    args = []
    if lang == 'cpp':
        name = 't.cpp'
        args.append('-std=c++11')
    elif lang == 'objc':
        name = 't.m'
    elif lang != 'c':
        raise Exception('Unknown language: %s' % lang)

    if all_warnings:
        args += ['-Wall', '-Wextra']

    index = Index.create()
    tu = index.parse(name, args=args, unsaved_files=[(name, source)])
    assert tu is not None
    return tu

def get_cursor(source, spelling):
    """Obtain a cursor from a source object.

    This provides a convenient search mechanism to find a cursor with specific
    spelling within a source. The first argument can be either a
    TranslationUnit or Cursor instance.

    If the cursor is not found, None is returned.
    """
    children = []
    if isinstance(source, Cursor):
        children = source.get_children()
    else:
        # Assume TU
        children = source.cursor.get_children()

    for cursor in children:
        if cursor.spelling == spelling:
            return cursor

        # Recurse into children.
        result = get_cursor(cursor, spelling)
        if result is not None:
            return result

    return None


__all__ = [
    'get_cursor',
    'get_tu',
]
