// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

namespace N {
  struct A { };
  namespace M { 
    struct C { };
  };
}

namespace N {
  struct B { };
}

// CHECK-CC1: A : 0
// CHECK-CC1: B : 0
// CHECK-CC1: M : 0
N::