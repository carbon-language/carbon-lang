// Note: the run lines follow their respective tests, since line/column
// matter in this test.

int f(int);

int test(int i, int j, int k, int l) {
  return i | j | k & l;
}

// RUN: c-index-test -code-completion-at=%s:7:9 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )}
// CHECK-CC1: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )}
// RUN: c-index-test -code-completion-at=%s:7:14 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:7:18 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:7:22 %s | FileCheck -check-prefix=CHECK-CC1 %s
