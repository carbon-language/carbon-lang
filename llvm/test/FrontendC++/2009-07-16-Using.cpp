// RUN: %llvmgxx %s -S

namespace A {
  typedef int B;
}
struct B {
};
using ::A::B;
