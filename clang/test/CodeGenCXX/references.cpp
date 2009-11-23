// RUN: clang-cc -verify -emit-llvm -o - %s | FileCheck %s

void t1() {
  extern int& a;
  int b = a; 
}

void t2(int& a) {
  int b = a;
}

int g;
int& gr = g;
int& grr = gr;
void t3() {
  int b = gr;
}

// Test reference binding.

struct C { int a; };

void f(const bool&);
void f(const int&);
void f(const _Complex int&);
void f(const C&);

C aggregate_return();

bool& bool_reference_return();
int& int_reference_return();
_Complex int& complex_int_reference_return();
C& aggregate_reference_return();

void test_bool() {
  bool a = true;
  f(a);

  f(true);
  
  bool_reference_return() = true;
  a = bool_reference_return();
}

void test_scalar() {
  int a = 10;
  f(a);
  
  struct { int bitfield : 3; } s = { 3 };
  f(s.bitfield);
  
  f(10);

  __attribute((vector_size(16))) typedef int vec4;
  f((vec4){1,2,3,4}[0]);
  
  int_reference_return() = 10;
  a = int_reference_return();
}

void test_complex() {
  _Complex int a = 10i;
  f(a);
  
  f(10i);
  
  complex_int_reference_return() = 10i;
  a = complex_int_reference_return();
}

void test_aggregate() {
  C c;
  f(c);

  f(aggregate_return());
  aggregate_reference_return().a = 10;

  c = aggregate_reference_return();
}

int& reference_return() {
  return g;
}

int reference_decl() {
  int& a = g;
  const int& b = 1;
  return a+b;
}

struct A {
  int& b();
};

void f(A* a) {
  int b = a->b();
}

// PR5122
void *foo = 0;
void * const & kFoo = foo;

struct D : C { D(); ~D(); };

void h() {
  // CHECK: call void @_ZN1DD1Ev
  const C& c = D();
}

namespace T {
  struct A {
    A();
    ~A();
  };

  struct B {
    B();
    ~B();
    A f();
  };

  void f() {
    // CHECK: call void @_ZN1T1BC1Ev
    // CHECK: call void @_ZN1T1B1fEv
    // CHECK: call void @_ZN1T1BD1Ev
    const A& a = B().f();
    // CHECK: call void @_ZN1T1fEv
    f();
    // CHECK: call void @_ZN1T1AD1Ev
  }
}

// PR5227.
namespace PR5227 {
void f(int &a) {
  (a = 10) = 20;
}
}

// PR5590
struct s0;
struct s1 { struct s0 &s0; };
void f0(s1 a) { s1 b = a; }
