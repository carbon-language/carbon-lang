namespace N4 {
  namespace N3 { }
}

class N3;

namespace N2 {
  namespace I1 { }
  namespace I4 = I1;
  namespace I5 { }
  namespace I1 { }
  
  namespace New =
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:13:18 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: I1 : 1
  // CHECK-CC1: I4 : 1
  // CHECK-CC1: I5 : 1
  // CHECK-CC1: N2 : 2
  // CHECK-NEXT-CC1: N4 : 2
  // RUN: true
  