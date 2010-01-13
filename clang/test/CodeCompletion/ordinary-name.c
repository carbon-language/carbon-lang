struct X { int x; };

typedef struct t TYPEDEF;

void foo() {
  int y;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:9 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: foo
  // CHECK-CC1: y
  // CHECK-CC1: TYPEDEF
