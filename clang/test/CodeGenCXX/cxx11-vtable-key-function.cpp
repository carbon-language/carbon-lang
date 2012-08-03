// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -std=c++11 | FileCheck %s
// PR13424

struct X {
  virtual ~X() = default;
  virtual void f();
};

void X::f() {}

// CHECK: @_ZTV1X = unnamed_addr constant
