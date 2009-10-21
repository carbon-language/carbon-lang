// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s
template <typename T> void f(T) {}
template <typename T> void f() { }

void test() {
  // CHECK: @_Z1fIiEvT_
  void (*p)(int) = &f;
  
  // CHECK: @_Z1fIiEvv
  void (*p2)() = f<int>;
}
// CHECK: define linkonce_odr void @_Z1fIiEvT_
// CHECK: define linkonce_odr void @_Z1fIiEvv
