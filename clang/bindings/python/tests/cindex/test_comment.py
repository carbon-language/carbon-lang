from clang.cindex import TranslationUnit
from tests.cindex.util import get_cursor

def test_comment():
    files = [('fake.c', """
/// Aaa.
int test1;

/// Bbb.
/// x
void test2(void);

void f() {

}
""")]
    # make a comment-aware TU
    tu = TranslationUnit.from_source('fake.c', ['-std=c99'], unsaved_files=files,
            options=TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION)
    test1 = get_cursor(tu, 'test1')
    assert test1 is not None, "Could not find test1."
    assert test1.type.is_pod()
    raw = test1.raw_comment
    brief = test1.brief_comment
    assert raw == """/// Aaa."""
    assert brief == """Aaa."""
    
    test2 = get_cursor(tu, 'test2')
    raw = test2.raw_comment
    brief = test2.brief_comment
    assert raw == """/// Bbb.\n/// x"""
    assert brief == """Bbb. x"""
    
    f = get_cursor(tu, 'f')
    raw = f.raw_comment
    brief = f.brief_comment
    assert raw is None
    assert brief is None


