// Note: the run lines follow their respective tests, since line/column
// matter in this test.

#define FOO(Arg1,Arg2) foobar
#define nil 0
void f() {

}

void g(int);

void f2() {
  int *ip = nil;
  ip = nil;
  g(nil);
}

// RUN: c-index-test -code-completion-at=%s:7:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:7:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: macro definition:{TypedText FOO}{LeftParen (}{Placeholder Arg1}{Comma , }{Placeholder Arg2}{RightParen )}
// RUN: c-index-test -code-completion-at=%s:13:13 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:14:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: macro definition:{TypedText nil} (30)
// RUN: c-index-test -code-completion-at=%s:15:5 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:15:5 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: macro definition:{TypedText nil} (60)
