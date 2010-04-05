// Note: the run lines follow their respective tests, since line/column
// matter in this test.

#define FOO(Arg1,Arg2) foobar

void f() {

}

// RUN: c-index-test -code-completion-at=%s:7:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: macro definition:{TypedText FOO}{LeftParen (}{Placeholder Arg1}{Comma , }{Placeholder Arg2}{RightParen )}
