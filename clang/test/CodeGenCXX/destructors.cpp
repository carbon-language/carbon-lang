// RUN: %clang_cc1 %s -emit-llvm -o - -mconstructor-aliases | FileCheck %s
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

// CHECK: @_ZN5test01AD1Ev = alias {{.*}} @_ZN5test01AD2Ev

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
