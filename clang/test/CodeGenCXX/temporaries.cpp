// RUN: clang-cc -emit-llvm %s -o %t -triple=x86_64-apple-darwin9 && 
struct A {
  A();
  ~A();
  void f();
};

// RUN: grep "call void @_ZN1AC1Ev" %t | count 2 &&
// RUN: grep "call void @_ZN1AD1Ev" %t | count 2 &&
void f1() {
  (void)A();
  A().f();
}

// Function calls
struct B {
  B();
  ~B();
};

B g();

// RUN: grep "call void @_ZN1BC1Ev" %t | count 0 &&
// RUN: grep "call void @_ZN1BD1Ev" %t | count 1 &&
void f2() {
  (void)g();
}

// Member function calls
struct C {
  C();
  ~C();
  
  C f();
};

// RUN: grep "call void @_ZN1CC1Ev" %t | count 1 &&
// RUN: grep "call void @_ZN1CD1Ev" %t | count 2 &&
void f3() {
  C().f();
}

// Function call operator
struct D {
  D();
  ~D();
  
  D operator()();
};

// RUN: grep "call void @_ZN1DC1Ev" %t | count 1 &&
// RUN: grep "call void @_ZN1DD1Ev" %t | count 2 &&
void f4() {
  D()();
}

// Overloaded operators
struct E {
  E();
  ~E();
  E operator+(const E&);
  E operator!();
};

// RUN: grep "call void @_ZN1EC1Ev" %t | count 3 &&
// RUN: grep "call void @_ZN1ED1Ev" %t | count 5 &&
void f5() {
  E() + E();
  !E();
}

struct F {
  F();
  ~F();
  F& f();
};

// RUN: grep "call void @_ZN1FC1Ev" %t | count 1 &&
// RUN: grep "call void @_ZN1FD1Ev" %t | count 1 
void f6() {
  F().f();
}

