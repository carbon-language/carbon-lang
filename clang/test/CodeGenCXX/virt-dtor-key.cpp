// RUN: clang-cc -emit-llvm %s -o - | FileCheck %s
// CHECK: @_ZTI3foo = constant
class foo {
   foo();
   virtual ~foo();
};

foo::~foo() {
}
