// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// CHECK: @_ZTI3foo = constant
class foo {
   foo();
   virtual ~foo();
};

foo::~foo() {
}
