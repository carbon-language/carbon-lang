// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// CHECK: @_ZTI3foo = constant
class foo {
   foo();
   virtual ~foo();
};

foo::~foo() {
}
