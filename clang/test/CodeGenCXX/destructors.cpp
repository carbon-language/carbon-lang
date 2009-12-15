// RUN: %clang_cc1 %s -emit-llvm -o -
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
