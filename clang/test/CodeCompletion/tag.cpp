class X { };
struct Y { };

namespace N {
  template<typename> class Z;
}

namespace M {
  class A;
}
using M::A;

namespace N {
  class Y;
  
  void test() {
    class
    // RUN: clang-cc -fsyntax-only -code-completion-at=%s:17:10 %s -o - | FileCheck -check-prefix=CC1 %s &&
    // CHECK-CC1: Y : 2
    // CHECK-CC1: Z : 2
    // CHECK-CC1: A : 4
    // CHECK-CC1: X : 4
    // CHECK-CC1: Y : 4
    // CHECK-CC1: M : 9
    // CHECK-CC1: N : 9
    // RUN: true
