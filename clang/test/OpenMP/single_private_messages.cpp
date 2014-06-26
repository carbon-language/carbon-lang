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
class S4 { // expected-note {{'S4' declared here}}
  int a;
  S4();

public:
  S4(int v) : a(v) {}
};
class S5 { // expected-note {{'S5' declared here}}
  int a;
  S5() : a(0) {}

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
  int &j = i;                // expected-note {{'j' defined here}}
#pragma omp single private // expected-error {{expected '(' after 'private'}}
  foo();
#pragma omp single private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp single private() // expected-error {{expected expression}}
  foo();
#pragma omp single private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp single private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp single private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  foo();
#pragma omp single private(argc)
  foo();
#pragma omp single private(S1) // expected-error {{'S1' does not refer to a value}}
  foo();
#pragma omp single private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
  foo();
#pragma omp single private(argv[1]) // expected-error {{expected variable name}}
  foo();
#pragma omp single private(e, g)
  foo();
#pragma omp single private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  foo();
#pragma omp single shared(i) // expected-error {{unexpected OpenMP clause 'shared' in directive '#pragma omp single'}}
  foo();
#pragma omp parallel
  {
    int v = 0;
    int i;
#pragma omp single private(i)
    foo();
    v += i;
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp single private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type}}
  foo();
#pragma omp single private(i)
  foo();
  return 0;
}

int main(int argc, char **argv) {
  S4 e(4); // expected-note {{'e' defined here}}
  S5 g(5); // expected-note {{'g' defined here}}
  int i;
  int &j = i;                // expected-note {{'j' defined here}}
#pragma omp single private // expected-error {{expected '(' after 'private'}}
  foo();
#pragma omp single private( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp single private() // expected-error {{expected expression}}
  foo();
#pragma omp single private(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp single private(argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp single private(argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  foo();
#pragma omp single private(argc)
  foo();
#pragma omp single private(S1) // expected-error {{'S1' does not refer to a value}}
  foo();
#pragma omp single private(a, b) // expected-error {{private variable with incomplete type 'S1'}}
  foo();
#pragma omp single private(argv[1]) // expected-error {{expected variable name}}
  foo();
#pragma omp single private(e, g) // expected-error 2 {{private variable must have an accessible, unambiguous default constructor}}
  foo();
#pragma omp single private(h) // expected-error {{threadprivate or thread local variable cannot be private}}
  foo();
#pragma omp single shared(i) // expected-error {{unexpected OpenMP clause 'shared' in directive '#pragma omp single'}}
  foo();
#pragma omp parallel
  {
    int i;
#pragma omp single private(i)
    foo();
  }
#pragma omp parallel shared(i)
#pragma omp parallel private(i)
#pragma omp single private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type}}
  foo();
#pragma omp single private(i)
  foo();

  return 0;
}

