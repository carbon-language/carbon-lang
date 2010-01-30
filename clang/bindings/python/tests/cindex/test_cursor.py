from clang.cindex import Index, CursorKind

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
    index = Index.create()
    tu = index.parse('t.c', unsaved_files = [('t.c',kInput)])
    
    # Skip until past start_decl.
    it = tu.cursor.get_children()
    while it.next().spelling != 'start_decl':
        pass

    tu_nodes = list(it)

    assert len(tu_nodes) == 3

    assert tu_nodes[0].kind == CursorKind.STRUCT_DECL
    assert tu_nodes[0].spelling == 's0'
    assert tu_nodes[0].is_definition() == True
    assert tu_nodes[0].location.file.name == 't.c'
    assert tu_nodes[0].location.line == 4
    assert tu_nodes[0].location.column == 8

    s0_nodes = list(tu_nodes[0].get_children())
    assert len(s0_nodes) == 2
    assert s0_nodes[0].kind == CursorKind.FIELD_DECL
    assert s0_nodes[0].spelling == 'a'
    assert s0_nodes[1].kind == CursorKind.FIELD_DECL
    assert s0_nodes[1].spelling == 'b'

    assert tu_nodes[1].kind == CursorKind.STRUCT_DECL
    assert tu_nodes[1].spelling == 's1'
    assert tu_nodes[1].is_definition() == False

    assert tu_nodes[2].kind == CursorKind.FUNCTION_DECL
    assert tu_nodes[2].spelling == 'f0'
    assert tu_nodes[2].is_definition() == True
