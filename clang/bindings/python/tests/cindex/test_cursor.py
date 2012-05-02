from clang.cindex import CursorKind
from clang.cindex import TypeKind
from .util import get_cursor
from .util import get_tu

kInput = """\
// FIXME: Find nicer way to drop builtins and other cruft.
int start_decl;

struct s0 {
  int a;
  int b;
};

struct s1;

void f0(int a0, int a1) {
  int l0, l1;

  if (a0)
    return;

  for (;;) {
    break;
  }
}
"""

def test_get_children():
    tu = get_tu(kInput)

    # Skip until past start_decl.
    it = tu.cursor.get_children()
    while it.next().spelling != 'start_decl':
        pass

    tu_nodes = list(it)

    assert len(tu_nodes) == 3

    assert tu_nodes[0] != tu_nodes[1]
    assert tu_nodes[0].kind == CursorKind.STRUCT_DECL
    assert tu_nodes[0].spelling == 's0'
    assert tu_nodes[0].is_definition() == True
    assert tu_nodes[0].location.file.name == 't.c'
    assert tu_nodes[0].location.line == 4
    assert tu_nodes[0].location.column == 8
    assert tu_nodes[0].hash > 0

    s0_nodes = list(tu_nodes[0].get_children())
    assert len(s0_nodes) == 2
    assert s0_nodes[0].kind == CursorKind.FIELD_DECL
    assert s0_nodes[0].spelling == 'a'
    assert s0_nodes[0].type.kind == TypeKind.INT
    assert s0_nodes[1].kind == CursorKind.FIELD_DECL
    assert s0_nodes[1].spelling == 'b'
    assert s0_nodes[1].type.kind == TypeKind.INT

    assert tu_nodes[1].kind == CursorKind.STRUCT_DECL
    assert tu_nodes[1].spelling == 's1'
    assert tu_nodes[1].displayname == 's1'
    assert tu_nodes[1].is_definition() == False

    assert tu_nodes[2].kind == CursorKind.FUNCTION_DECL
    assert tu_nodes[2].spelling == 'f0'
    assert tu_nodes[2].displayname == 'f0(int, int)'
    assert tu_nodes[2].is_definition() == True

def test_underlying_type():
    tu = get_tu('typedef int foo;')
    typedef = get_cursor(tu, 'foo')
    assert typedef is not None

    assert typedef.kind.is_declaration()
    underlying = typedef.underlying_typedef_type
    assert underlying.kind == TypeKind.INT

def test_enum_type():
    tu = get_tu('enum TEST { FOO=1, BAR=2 };')
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL
    enum_type = enum.enum_type
    assert enum_type.kind == TypeKind.UINT

def test_enum_type_cpp():
    tu = get_tu('enum TEST : long long { FOO=1, BAR=2 };', lang="cpp")
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL
    assert enum.enum_type.kind == TypeKind.LONGLONG

def test_objc_type_encoding():
    tu = get_tu('int i;', lang='objc')
    i = get_cursor(tu, 'i')

    assert i is not None
    assert i.objc_type_encoding == 'i'

def test_enum_values():
    tu = get_tu('enum TEST { SPAM=1, EGG, HAM = EGG * 20};')
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL

    enum_constants = list(enum.get_children())
    assert len(enum_constants) == 3

    spam, egg, ham = enum_constants

    assert spam.kind == CursorKind.ENUM_CONSTANT_DECL
    assert spam.enum_value == 1
    assert egg.kind == CursorKind.ENUM_CONSTANT_DECL
    assert egg.enum_value == 2
    assert ham.kind == CursorKind.ENUM_CONSTANT_DECL
    assert ham.enum_value == 40

def test_enum_values_cpp():
    tu = get_tu('enum TEST : long long { SPAM = -1, HAM = 0x10000000000};', lang="cpp")
    enum = get_cursor(tu, 'TEST')
    assert enum is not None

    assert enum.kind == CursorKind.ENUM_DECL

    enum_constants = list(enum.get_children())
    assert len(enum_constants) == 2

    spam, ham = enum_constants

    assert spam.kind == CursorKind.ENUM_CONSTANT_DECL
    assert spam.enum_value == -1
    assert ham.kind == CursorKind.ENUM_CONSTANT_DECL
    assert ham.enum_value == 0x10000000000

def test_annotation_attribute():
    tu = get_tu('int foo (void) __attribute__ ((annotate("here be annotation attribute")));')

    foo = get_cursor(tu, 'foo')
    assert foo is not None

    for c in foo.get_children():
        if c.kind == CursorKind.ANNOTATE_ATTR:
            assert c.displayname == "here be annotation attribute"
            break
    else:
        assert False, "Couldn't find annotation"
