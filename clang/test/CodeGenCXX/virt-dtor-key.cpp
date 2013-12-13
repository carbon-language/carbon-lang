// RUN: %clang_cc1 -cxx-abi itanium -emit-llvm %s -o - | FileCheck %s
// CHECK: @_ZTI3foo = unnamed_addr constant
class foo {
   foo();
   virtual ~foo();
};

foo::~foo() {
}
