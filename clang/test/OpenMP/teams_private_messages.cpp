// RUN: %clang_cc1 -verify -fopenmp=libiomp5 %s

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
class S4 {
  int a;
  S4(); // expected-note {{implicitly declared private here}}
public:
  S4(int v):a(v) { }
};
class S5 {
  int a;
  S5():a(0) {} // expected-note {{implicitly declared private here}}
public:
  S5(int v):a(v) { }
};

int threadvar;
#pragma omp threadprivate(threadvar) // expected-note {{defined as threadprivate or thread local}}

int main(int argc, char **argv) {
  const int d = 5; // expected-note {{constant variable is predetermined as shared}}
  const int da[5] = { 0 }; // expected-note {{constant variable is predetermined as shared}}
  S4 e(4);
  S5 g(5);
  int i;
  int &j = i; // expected-note {{'j' defined here}}
  #pragma omp target
  #pragma omp teams private // expected-error {{expected '(' after 'private'}}
  foo();
  #pragma omp target
  #pragma omp teams private ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target
  #pragma omp teams private () // expected-error {{expected expression}}
  foo();
  #pragma omp target
  #pragma omp teams private (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target
  #pragma omp teams private (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp target
  #pragma omp teams private (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  foo();
  #pragma omp target
  #pragma omp teams private (argc argv) // expected-error {{expected ',' or ')' in 'private' clause}}
  foo();
  #pragma omp target
  #pragma omp teams private (S1) // expected-error {{'S1' does not refer to a value}}
  foo();
  #pragma omp target
  #pragma omp teams private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}} expected-error 3 {{shared variable cannot be private}}
  foo();
  #pragma omp target
  #pragma omp teams private (argv[1]) // expected-error {{expected variable name}}
  foo();
  #pragma omp target
  #pragma omp teams private(ba)
  foo();
  #pragma omp target
  #pragma omp teams private(ca) // expected-error {{shared variable cannot be private}}
  foo();
  #pragma omp target
  #pragma omp teams private(da) // expected-error {{shared variable cannot be private}}
  foo();
  #pragma omp target
  #pragma omp teams private(S2::S2s) // expected-error {{shared variable cannot be private}}
  foo();
  #pragma omp target
  #pragma omp teams private(e, g) // expected-error {{calling a private constructor of class 'S4'}} expected-error {{calling a private constructor of class 'S5'}}
  foo();
  #pragma omp target
  #pragma omp teams private(threadvar) // expected-error {{threadprivate or thread local variable cannot be private}}
  foo();
  #pragma omp target
  #pragma omp teams shared(i), private(i) // expected-error {{shared variable cannot be private}} expected-note {{defined as shared}}
  foo();
  #pragma omp target
  #pragma omp teams firstprivate(i) private(i) // expected-error {{firstprivate variable cannot be private}} expected-note {{defined as firstprivate}}
  foo();
  #pragma omp target
  #pragma omp teams private(i)
  foo();
  #pragma omp target
  #pragma omp teams private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type 'int &'}}
  foo();
  #pragma omp target
  #pragma omp teams firstprivate(i)
  for (int k = 0; k < 10; ++k) {
    #pragma omp parallel private(i)
    foo();
  }

  return 0;
}
