namespace N {
  struct A { };
  namespace M { 
    struct C { };
  };
}

namespace N {
  struct B { };
}

N::
// RUN: clang-cc -fsyntax-only -code-completion-at=%s:12:4 %s -o - | FileCheck -check-prefix=CC1 %s &&
// CHECK-CC1: A : 0
// CHECK-CC1: B : 0
// CHECK-CC1: M : 0
// RUN: true

