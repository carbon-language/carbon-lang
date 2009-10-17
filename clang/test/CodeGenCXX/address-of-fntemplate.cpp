// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s
template <typename T> void f(T) {}

void test() {
  // CHECK: @_Z1fIiEvT_
  void (*p)(int) = &f;
}
// CHECK: define linkonce_odr void @_Z1fIiEvT_
