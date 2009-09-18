// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

namespace N3 {
}

namespace N2 {
  namespace I1 { }
  namespace I4 = I1;
  namespace I5 { }
  namespace I1 { }
  
  // CHECK-CC1: I1 : 0
  // CHECK-NEXT-CC1: I5 : 0
  namespace
