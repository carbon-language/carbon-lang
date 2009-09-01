// RUN: clang-cc -verify -emit-llvm -o %t %s

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
