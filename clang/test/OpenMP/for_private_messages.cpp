// RUN: %clang_cc1 -verify -fopenmp=libiomp5 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}} expected-note 2 {{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;

public:
  S2() : a(0) {}
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;

public:
  S3() : a(0) {}
};
const S3 ca[5];
class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}}

public:
  S4(int v) : a(v) {}
};
class S5 {
  int a;
  S5() : a(0) {} // expected-note {{implicitly declared private here}}

public:
  S5(int v) : a(v) {}
};

S3 h;
#pragma omp threadprivate(h) // expected-note 2 {{defined as threadprivate or thread local}}

template <class I, class C>
int foomain(I argc, C **argv) {
  I e(4);
  I g(5);
  int i;
  int &j = i;           // expected-note {{'j' defined here}}
#pragma omp for private // expected-error {{expected '(' after 'private'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(e, g)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for shared(i) // expected-error {{unexpected OpenMP clause 'shared' in directive '#pragma omp for'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp for private(i)
    for (int k = 0; k < argc; ++k) {
      i = k;
      v += i;
    }
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp for private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(i)
  for (int k = 0; k < argc; ++k)
    ++k;
  return 0;
}

namespace A {
double x;
#pragma omp threadprivate(x) // expected-note {{defined as threadprivate or thread local}}
}
namespace B {
using A::x;
}

int main(int argc, char **argv) {
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i;           // expected-note {{'j' defined here}}
#pragma omp for private // expected-error {{expected '(' after 'private'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private() // expected-error {{expected expression}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argc)
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(S1) // expected-error {{'S1' does not refer to a value}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(argv[1]) // expected-error {{expected variable name}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(B::x) // expected-error {{threadprivate or thread local variable cannot be private}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for shared(i) // expected-error {{unexpected OpenMP clause 'shared' in directive '#pragma omp for'}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp parallel
  {
    int i;
#pragma omp for private(i)
    for (int k = 0; k < argc; ++k)
      ++k;
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp for private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type}}
  for (int k = 0; k < argc; ++k)
    ++k;
#pragma omp for private(i)
  for (int k = 0; k < argc; ++k)
    ++k;

  return 0;
}

