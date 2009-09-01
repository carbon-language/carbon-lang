// RUN: clang-cc %s -emit-llvm -o -
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