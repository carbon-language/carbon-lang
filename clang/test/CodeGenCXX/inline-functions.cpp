// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK: ; ModuleID 

struct A {
    inline void f();
};

// CHECK-NOT: define void @_ZN1A1fEv
void A::f() { }

template<typename> struct B { };

template<> struct B<char> {
  inline void f();
};

// CHECK-NOT: _ZN1BIcE1fEv
void B<char>::f() { }

// We need a final CHECK line here.

// CHECK: define void @_Z1fv
void f() { }
