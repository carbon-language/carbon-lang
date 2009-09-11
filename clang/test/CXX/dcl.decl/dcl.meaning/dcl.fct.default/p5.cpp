// RUN: clang-cc -fsyntax-only -verify %s

float global_f;

void f0(int *ip = &global_f); // expected-error{{incompatible}}

// Example from C++03 standard
int a = 1; 
int f(int); 
int g(int x = f(a));

void h() { 
  a = 2;
  {
    int *a = 0;
    g(); // FIXME: check that a is called with a value of 2
  }
}
