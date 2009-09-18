// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

class T { };

typedef int Integer;

namespace N { }

void f() {
  typedef float Float;
  
  // CHECK-CC1: Float : 0
  // CHECK-CC1: + : 0
  // CHECK-CC1: short : 0
  // CHECK-CC1: Integer : 2
  // CHECK-CC1: T : 2
  // CHECK-CC1: N : 5
  operator
