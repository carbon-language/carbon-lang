// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}} expected-note{{forward declaration of 'S1'}}
extern S1 a;
class S2 {
  mutable int a;
public:
  S2():a(0) { }
  static float S2s; // expected-note {{static data member is predetermined as shared}}
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 c; // expected-note {{global variable is predetermined as shared}}
const S3 ca[5]; // expected-note {{global variable is predetermined as shared}}
extern const int f; // expected-note {{global variable is predetermined as shared}}
class S4 { // expected-note {{'S4' declared here}}
  int a;
  S4();
public:
  S4(int v):a(v) { }
};
class S5 { // expected-note {{'S5' declared here}}
  int a;
  S5():a(0) {}
public:
  S5(int v):a(v) { }
};

int threadvar;
#pragma omp threadprivate(threadvar) // expected-note {{defined as threadprivate or thread local}}

int main(int argc, char **argv) {
  const int d = 5; // expected-note {{constant variable is predetermined as shared}}
  const int da[5] = { 0 }; // expected-note {{constant variable is predetermined as shared}}
  S4 e(4); // expected-note {{'e' defined here}}
  S5 g(5); // expected-note {{'g' defined here}}
  int i;
  int &j = i; // expected-note {{'j' defined here}}
  #pragma omp parallel private // expected-error {{expected '(' after 'private'}}
  #pragma omp parallel private ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel private () // expected-error {{expected expression}}
  #pragma omp parallel private (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel private (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel private (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  #pragma omp parallel private (argc argv) // expected-error {{expected ',' or ')' in 'private' clause}}
  #pragma omp parallel private (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp parallel private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}} expected-error 3 {{shared variable cannot be private}}
  #pragma omp parallel private (argv[1]) // expected-error {{expected variable name}}
  #pragma omp parallel private(ba)
  #pragma omp parallel private(ca) // expected-error {{shared variable cannot be private}}
  #pragma omp parallel private(da) // expected-error {{shared variable cannot be private}}
  #pragma omp parallel private(S2::S2s) // expected-error {{shared variable cannot be private}}
  #pragma omp parallel private(e, g) // expected-error 2 {{private variable must have an accessible, unambiguous default constructor}}
  #pragma omp parallel private(threadvar) // expected-error {{threadprivate or thread local variable cannot be private}}
  #pragma omp parallel shared(i), private(i) // expected-error {{shared variable cannot be private}} expected-note {{defined as shared}}
  foo();
  #pragma omp parallel firstprivate(i) private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  foo();
  #pragma omp parallel private(i)
  #pragma omp parallel private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type 'int &'}}
  foo();
  #pragma omp parallel firstprivate(i)
  for (int k = 0; k < 10; ++k) {
    #pragma omp parallel private(i)
    foo();
  }

  return 0;
}
