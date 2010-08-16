// Note: the run lines follow their respective tests, since line/column
// matter in this test.

int f(int);

int test(int i, int j, int k, int l) {
  return i | j | k & l;
}

struct X f1 = { 17 };
void f2() { f1(17); }

const char *str = "Hello, \nWorld";

// RUN: c-index-test -code-completion-at=%s:7:9 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:7:9 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: macro definition:{TypedText __VERSION__} (70)
// CHECK-CC1: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (12)
// CHECK-CC1-NOT: NotImplemented:{TypedText float} (40)
// CHECK-CC1: ParmDecl:{ResultType int}{TypedText j} (2)
// CHECK-CC1: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (30)
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:7:9 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1a %s
// CHECK-CC1a: ParmDecl:{ResultType int}{TypedText j} (2)
// CHECK-CC1a: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (30)
// CHECK-CC1a: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (12)
// CHECK-CC1a: macro definition:{TypedText __VERSION__} (70)
// RUN: c-index-test -code-completion-at=%s:7:14 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:7:14 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: macro definition:{TypedText __VERSION__} (70)
// CHECK-CC3: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC3-NOT: NotImplemented:{TypedText float} (40)
// CHECK-CC3: ParmDecl:{ResultType int}{TypedText j} (8)
// CHECK-CC3: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expressio

// RUN: c-index-test -code-completion-at=%s:7:18 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:7:22 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: c-index-test -code-completion-at=%s:7:2 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: macro definition:{TypedText __VERSION__} (70)
// CHECK-CC2: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC2: NotImplemented:{TypedText float} (40)
// CHECK-CC2: ParmDecl:{ResultType int}{TypedText j} (8)
// CHECK-CC2: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (30)
// RUN: c-index-test -code-completion-at=%s:11:16 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC4: VarDecl:{ResultType struct X}{TypedText f1} (50)

// RUN: c-index-test -code-completion-at=%s:13:28 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: NotImplemented:{TypedText void} (40)
// CHECK-CC5: NotImplemented:{TypedText volatile} (40)
