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
void t3() {
  int b = gr;
}

// Test reference binding.

struct C {};

void f(const bool&);
void f(const int&);
void f(const _Complex int&);
void f(const C&);

C structfunc();

void test_bool() {
  bool a = true;
  f(a);

  f(true);
}

void test_scalar() {
  int a = 10;
  f(a);
  
  struct { int bitfield : 3; } s = { 3 };
  f(s.bitfield);
  
  f(10);

  __attribute((vector_size(16))) typedef int vec4;
  f((vec4){1,2,3,4}[0]);
}

void test_complex() {
  _Complex int a = 10i;
  f(a);
  
  f(10i);
}

void test_aggregate() {
  C c;
  f(c);

  f(structfunc());
}

