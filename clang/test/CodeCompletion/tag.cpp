// RUN: clang-cc -fsyntax-only -code-completion-dump=1 %s -o - | FileCheck -check-prefix=CC1 %s &&
// RUN: true

class X { };
struct Y { };

namespace N {
  template<typename> class Z;
}

namespace N {
  class Y;
  
  void test() {
    // CHECK-CC1: Y : 2
    // CHECK-CC1: Z : 2
    // CHECK-CC1: X : 3
    // CHECK-CC1: Y : 3
    // CHECK-CC1: N : 6
    class
