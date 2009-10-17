// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s
template <typename T> void f(T) {}

void test() {
  // FIXME: This emits only a declaration instead of a definition
  // CHECK: @_Z1fIiEvT_
  void (*p)(int) = &f;
}
// CHECK-disabled: define linkonce_odr void @_Z1fIiEvT_
