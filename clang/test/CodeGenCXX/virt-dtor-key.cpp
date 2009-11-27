// RUN: clang-cc -emit-llvm %s -o - | FileCheck %s
// CHECK: @_ZTI3foo = linkonce_odr constant
class foo {
   foo();
   virtual ~foo();
};

foo::~foo() {
}
