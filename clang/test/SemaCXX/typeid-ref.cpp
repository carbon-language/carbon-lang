// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
namespace std {
  class type_info;
}

struct X { };

void f() {
  // CHECK: @_ZTS1X = weak_odr constant
  // CHECK: @_ZTI1X = weak_odr constant 
  (void)typeid(X&);
}
