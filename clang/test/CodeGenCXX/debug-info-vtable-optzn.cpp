// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s
//
// This tests that the "emit debug info for a C++ class only in the
// module that has its vtable" optimization is disabled by default on
// Darwin.
//
// CHECK: [ DW_TAG_member ] [lost]
class A
{
  virtual bool f() = 0;
  int lost;
};

class B : public A
{
  B *g();
};

B *B::g() {
  return this;
}
