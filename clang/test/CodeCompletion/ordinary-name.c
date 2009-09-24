struct X { int x; };

typedef struct X TYPEDEF;

void foo() {
  int y;
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:6:9 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: y : 0
  // CHECK-CC1-NEXT: TYPEDEF : 2
  // CHECK-CC1-NOT: X
  // CHECK-CC1: foo : 2
  // RUN: true
