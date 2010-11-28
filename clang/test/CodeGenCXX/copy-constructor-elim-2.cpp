// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct A { int x; A(int); ~A(); };
A f() { return A(0); }
// CHECK: define void @_Z1fv
// CHECK: call void @_ZN1AC1Ei
// CHECK-NEXT: ret void

// Verify that we do not elide copies when constructing a base class.
namespace no_elide_base {
  struct Base { 
    Base(const Base&);
    ~Base();
  };

  struct Other {
    operator Base() const;
  };

  struct Derived : public virtual Base { 
    Derived(const Other &O);
  };

  // CHECK: define void @_ZN13no_elide_base7DerivedC1ERKNS_5OtherE
  Derived::Derived(const Other &O) 
    // CHECK: call void @_ZNK13no_elide_base5OthercvNS_4BaseEEv
    // CHECK: call void @_ZN13no_elide_base4BaseC2ERKS0_
    // CHECK: call void @_ZN13no_elide_base4BaseD1Ev
    : Base(O)
  {
    // CHECK: ret void
  }
}

// PR8683.

namespace PR8683 {

struct A {
  A();
  A(const A&);
  A& operator=(const A&);
};

struct B {
  A a;
};

void f() {
  // Verify that we don't mark the copy constructor in this expression as elidable.
  // CHECK: call void @_ZN6PR86831AC1ERKS0_
  A a = (B().a);
}

}
