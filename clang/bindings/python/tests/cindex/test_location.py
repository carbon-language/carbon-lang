from clang.cindex import Index

baseInput="int one;\nint two;\n"

def assert_location(loc, line, column, offset):
    assert loc.line == line
    assert loc.column == column
    assert loc.offset == offset

def test_location():
    index = Index.create()
    tu = index.parse('t.c', unsaved_files = [('t.c',baseInput)])

    for n in tu.cursor.get_children():
        if n.spelling == 'one':
            assert_location(n.location,line=1,column=5,offset=4)
        if n.spelling == 'two':
            assert_location(n.location,line=2,column=5,offset=13)

    # adding a linebreak at top should keep columns same
    tu = index.parse('t.c', unsaved_files = [('t.c',"\n"+baseInput)])

    for n in tu.cursor.get_children():
        if n.spelling == 'one':
            assert_location(n.location,line=2,column=5,offset=5)
        if n.spelling == 'two':
            assert_location(n.location,line=3,column=5,offset=14)

    # adding a space should affect column on first line only
    tu = index.parse('t.c', unsaved_files = [('t.c'," "+baseInput)])

    for n in tu.cursor.get_children():
        if n.spelling == 'one':
            assert_location(n.location,line=1,column=6,offset=5)
        if n.spelling == 'two':
            assert_location(n.location,line=2,column=5,offset=14)

def test_extent():
    index = Index.create()
    tu = index.parse('t.c', unsaved_files = [('t.c',baseInput)])

    for n in tu.cursor.get_children():
        if n.spelling == 'one':
            assert_location(n.extent.start,line=1,column=1,offset=0)
            assert_location(n.extent.end,line=1,column=8,offset=7)
            assert baseInput[n.extent.start.offset:n.extent.end.offset] == "int one"
        if n.spelling == 'two':
            assert_location(n.extent.start,line=2,column=1,offset=9)
            assert_location(n.extent.end,line=2,column=8,offset=16)
            assert baseInput[n.extent.start.offset:n.extent.end.offset] == "int two"
