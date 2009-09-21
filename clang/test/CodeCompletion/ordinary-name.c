// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

struct X { int x; };

typedef struct X TYPEDEF;

void foo() {
  int y;
  // CHECK-CC1: y : 0
  // CHECK-NEXT-CC1: TYPEDEF : 2
  // CHECK-NEXT-CC1: foo : 2
