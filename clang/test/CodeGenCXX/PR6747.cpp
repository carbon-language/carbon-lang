// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct foo {
  virtual void bar();
// CHECK: define available_externally void @_ZN3foo3bazEv
  virtual void baz() {}
};
void zed() {
  foo b;
  b.baz();
}
