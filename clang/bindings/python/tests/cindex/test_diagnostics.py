from clang.cindex import *

def tu_from_source(source):
    index = Index.create()
    tu = index.parse('INPUT.c', unsaved_files = [('INPUT.c', source)])
    return tu

# FIXME: We need support for invalid translation units to test better.

def test_diagnostic_warning():
    tu = tu_from_source("""int f0() {}\n""")
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 1
    assert tu.diagnostics[0].location.column == 11
    assert (tu.diagnostics[0].spelling ==
            'control reaches end of non-void function')

def test_diagnostic_note():
    # FIXME: We aren't getting notes here for some reason.
    index = Index.create()
    tu = tu_from_source("""#define A x\nvoid *A = 1;\n""")
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 2
    assert tu.diagnostics[0].location.column == 7
    assert 'incompatible' in tu.diagnostics[0].spelling
#    assert tu.diagnostics[1].severity == Diagnostic.Note
#    assert tu.diagnostics[1].location.line == 1
#    assert tu.diagnostics[1].location.column == 11
#    assert tu.diagnostics[1].spelling == 'instantiated from'

def test_diagnostic_fixit():
    index = Index.create()
    tu = tu_from_source("""struct { int f0; } x = { f0 : 1 };""")
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 1
    assert tu.diagnostics[0].location.column == 26
    assert tu.diagnostics[0].spelling.startswith('use of GNU old-style')
    assert len(tu.diagnostics[0].fixits) == 1
    assert tu.diagnostics[0].fixits[0].range.start.line == 1
    assert tu.diagnostics[0].fixits[0].range.start.column == 26
    assert tu.diagnostics[0].fixits[0].range.end.line == 1
    assert tu.diagnostics[0].fixits[0].range.end.column == 30
    assert tu.diagnostics[0].fixits[0].value == '.f0 = '

def test_diagnostic_range():
    index = Index.create()
    tu = tu_from_source("""void f() { int i = "a" + 1; }""")
    assert len(tu.diagnostics) == 1
    assert tu.diagnostics[0].severity == Diagnostic.Warning
    assert tu.diagnostics[0].location.line == 1
    assert tu.diagnostics[0].location.column == 16
    assert tu.diagnostics[0].spelling.startswith('incompatible pointer to')
    assert len(tu.diagnostics[0].fixits) == 0
    assert len(tu.diagnostics[0].ranges) == 1
    assert tu.diagnostics[0].ranges[0].start.line == 1
    assert tu.diagnostics[0].ranges[0].start.column == 20
    assert tu.diagnostics[0].ranges[0].end.line == 1
    assert tu.diagnostics[0].ranges[0].end.column == 27
    try:
      tu.diagnostics[0].ranges[1].start.line
    except IndexError:
      assert True
    else:
      assert False
      

