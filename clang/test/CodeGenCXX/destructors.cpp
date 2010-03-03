// RUN: %clang_cc1 %s -emit-llvm -o - -mconstructor-aliases | FileCheck %s

// CHECK: @_ZN5test01AD1Ev = alias {{.*}} @_ZN5test01AD2Ev
// CHECK: @_ZN5test11MD2Ev = alias {{.*}} @_ZN5test11AD2Ev
// CHECK: @_ZN5test11ND2Ev = alias {{.*}} @_ZN5test11AD2Ev
// CHECK: @_ZN5test11OD2Ev = alias {{.*}} @_ZN5test11AD2Ev
// CHECK: @_ZN5test11SD2Ev = alias bitcast {{.*}} @_ZN5test11AD2Ev

struct A {
  int a;
  
  ~A();
};

// Base with non-trivial destructor
struct B : A {
  ~B();
};

B::~B() { }

// Field with non-trivial destructor
struct C {
  A a;
  
  ~C();
};

C::~C() { }

// PR5084
template<typename T>
class A1 {
  ~A1();
};

template<> A1<char>::~A1();

// PR5529
namespace PR5529 {
  struct A {
    ~A();
  };
  
  A::~A() { }
  struct B : A {
    virtual ~B();
  };
  
  B::~B()  {}
}

// FIXME: there's a known problem in the codegen here where, if one
// destructor throws, the remaining destructors aren't run.  Fix it,
// then make this code check for it.
namespace test0 {
  void foo();
  struct VBase { ~VBase(); };
  struct Base { ~Base(); };
  struct Member { ~Member(); };

  struct A : Base {
    Member M;
    ~A();
  };

  // The function-try-block won't suppress -mconstructor-aliases here.
  A::~A() try { } catch (int i) {}

// complete destructor alias tested above

// CHECK: define void @_ZN5test01AD2Ev
// CHECK: invoke void @_ZN5test06MemberD1Ev
// CHECK:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test04BaseD2Ev
// CHECK:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]

  struct B : Base, virtual VBase {
    Member M;
    ~B();
  };
  B::~B() try { } catch (int i) {}
  // It will suppress the delegation optimization here, though.

// CHECK: define void @_ZN5test01BD1Ev
// CHECK: invoke void @_ZN5test06MemberD1Ev
// CHECK:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test04BaseD2Ev
// CHECK:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test05VBaseD2Ev
// CHECK:   unwind label [[VBASE_UNWIND:%[a-zA-Z0-9.]+]]

// CHECK: define void @_ZN5test01BD2Ev
// CHECK: invoke void @_ZN5test06MemberD1Ev
// CHECK:   unwind label [[MEM_UNWIND:%[a-zA-Z0-9.]+]]
// CHECK: invoke void @_ZN5test04BaseD2Ev
// CHECK:   unwind label [[BASE_UNWIND:%[a-zA-Z0-9.]+]]
}

// Test base-class aliasing.
namespace test1 {
  struct A { ~A(); char ***m; }; // non-trivial destructor
  struct B { ~B(); }; // non-trivial destructor
  struct Empty { }; // trivial destructor, empty
  struct NonEmpty { int x; }; // trivial destructor, non-empty

  // There must be a definition in this translation unit for the alias
  // optimization to apply.
  A::~A() { delete m; }

  struct M : A { ~M(); };
  M::~M() {} // alias tested above

  struct N : A, Empty { ~N(); };
  N::~N() {} // alias tested above

  struct O : Empty, A { ~O(); };
  O::~O() {} // alias tested above

  struct P : NonEmpty, A { ~P(); };
  P::~P() {} // CHECK: define void @_ZN5test11PD2Ev

  struct Q : A, B { ~Q(); };
  Q::~Q() {} // CHECK: define void @_ZN5test11QD2Ev

  struct R : A { ~R(); };
  R::~R() { A a; } // CHECK: define void @_ZN5test11RD2Ev

  struct S : A { ~S(); int x; };
  S::~S() {} // alias tested above

  struct T : A { ~T(); B x; };
  T::~T() {} // CHECK: define void @_ZN5test11TD2Ev

  // The VTT parameter prevents this.  We could still make this work
  // for calling conventions that are safe against extra parameters.
  struct U : A, virtual B { ~U(); };
  U::~U() {} // CHECK: define void @_ZN5test11UD2Ev
}

// PR6471
namespace test2 {
  struct A { ~A(); char ***m; };
  struct B : A { ~B(); };

  B::~B() {}
  // CHECK: define void @_ZN5test21BD2Ev
  // CHECK: call void @_ZN5test21AD2Ev
}
