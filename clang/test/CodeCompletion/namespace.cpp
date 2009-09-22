namespace N3 {
}

namespace N2 {
  namespace I1 { }
  namespace I4 = I1;
  namespace I5 { }
  namespace I1 { }
  
  namespace
  // RUN: clang-cc -fsyntax-only -code-completion-at=%s:10:12 %s -o - | FileCheck -check-prefix=CC1 %s &&
  // CHECK-CC1: I1 : 0
  // CHECK-NEXT-CC1: I5 : 0
  // RUN: true
  
