from clang.cindex import TranslationUnit

def test_code_complete():
    files = [('fake.c', """
/// Aaa.
int test1;

/// Bbb.
void test2(void);

void f() {

}
""")]

    tu = TranslationUnit.from_source('fake.c', ['-std=c99'], unsaved_files=files,
            options=TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION)

    cr = tu.codeComplete('fake.c', 9, 1, unsaved_files=files, include_brief_comments=True)
    assert cr is not None
    assert len(cr.diagnostics) == 0

    completions = []
    for c in cr.results:
        completions.append(str(c))

    expected = [
      "{'int', ResultType} | {'test1', TypedText} || Priority: 50 || Availability: Available || Brief comment: Aaa.",
      "{'void', ResultType} | {'test2', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 50 || Availability: Available || Brief comment: Bbb.",
      "{'return', TypedText} || Priority: 40 || Availability: Available || Brief comment: None"
    ]

    for c in expected:
        assert c in completions

