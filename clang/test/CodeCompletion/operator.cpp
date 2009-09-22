class T { };

typedef int Integer;

namespace N { }

void f() {
  typedef float Float;
  
  operator
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:10:11 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: Float : 0
  // CHECK-CC1: + : 0
  // CHECK-CC1: short : 0
  // CHECK-CC1: Integer : 2
  // CHECK-CC1: T : 2
  // CHECK-CC1: N : 5
  // RUN: true
