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

void test_bool() {
  bool a = true;
  f(a);

  f(true);
}

void test_scalar() {
  int a = 10;
  f(a);
  
  f(10);
}

void test_complex() {
  _Complex int a = 10i;
  f(a);
}

void test_aggregate() {
  C c;
  f(c);
}

