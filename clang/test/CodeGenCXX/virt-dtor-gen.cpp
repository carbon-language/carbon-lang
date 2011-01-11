// RUN: %clang_cc1 -o - -emit-llvm %s | FileCheck %s
// PR5483

// Make sure we generate all three forms of the destructor when it is virtual.
class Foo {
  virtual ~Foo();
};
Foo::~Foo() {}

// CHECK: define unnamed_addr void @_ZN3FooD0Ev
