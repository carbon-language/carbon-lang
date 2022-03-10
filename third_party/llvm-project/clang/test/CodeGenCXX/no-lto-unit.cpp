// RUN: %clang_cc1 -flto=thin -triple x86_64-unknown-linux -fvisibility hidden -emit-llvm-bc -o %t %s
// RUN: llvm-dis -o - %t | FileCheck %s
// RUN: %clang_cc1 -flto=thin -flto-unit -fno-lto-unit -triple x86_64-unknown-linux -fvisibility hidden -emit-llvm-bc -o %t %s
// RUN: llvm-dis -o - %t | FileCheck %s
// RUN: llvm-bcanalyzer -dump %t | FileCheck %s --check-prefix=NOLTOUNIT
// NOLTOUNIT: <FLAGS op0=0/>

// CHECK-NOT: !type
class A {
  virtual void f() {}
};

A *f() {
  return new A;
}
