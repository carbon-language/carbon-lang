// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

namespace N4 {
  namespace N3 { }
}

class N3;

namespace N2 {
  namespace I1 { }
  namespace I4 = I1;
  namespace I5 { }
  namespace I1 { }
  
  void foo() {
    // CHECK-CC1: I1 : 2
    // CHECK-CC1: I4 : 2
    // CHECK-CC1: I5 : 2
    // CHECK-CC1: N2 : 3
    // CHECK-NEXT-CC1: N4 : 3
    using namespace


