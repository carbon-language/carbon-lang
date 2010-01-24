from clang.cindex import CursorKind

def test_name():
    assert CursorKind.UNEXPOSED_DECL.name is 'UNEXPOSED_DECL'

def test_get_all_kinds():
    assert CursorKind.UNEXPOSED_DECL in CursorKind.get_all_kinds()
    assert CursorKind.TRANSLATION_UNIT in CursorKind.get_all_kinds()

def test_kind_groups():
    """Check that every kind classifies to exactly one group."""

    assert CursorKind.UNEXPOSED_DECL.is_declaration()
    assert CursorKind.TYPE_REF.is_reference()
    assert CursorKind.DECL_REF_EXPR.is_expression()
    assert CursorKind.UNEXPOSED_STMT.is_statement()
    assert CursorKind.INVALID_FILE.is_invalid()

    for k in CursorKind.get_all_kinds():
        group = [n for n in ('is_declaration', 'is_reference', 'is_expression',
                             'is_statement', 'is_invalid')
                 if getattr(k, n)()]

        if k == CursorKind.TRANSLATION_UNIT:
            assert len(group) == 0
        else:
            assert len(group) == 1
