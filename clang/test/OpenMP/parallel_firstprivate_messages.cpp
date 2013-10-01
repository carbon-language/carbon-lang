// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

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
  S2(S2 &s2):a(s2.a) { }
  static float S2s;
  static const float S2sc;
};
const float S2::S2sc = 0;
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
  S3(S3 &s3):a(s3.a) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
class S4 { // expected-note {{'S4' declared here}}
  int a;
  S4();
  S4(const S4 &s4);
public:
  S4(int v):a(v) { }
};
class S5 { // expected-note {{'S5' declared here}}
  int a;
  S5():a(0) {}
  S5(const S5 &s5):a(s5.a) { }
public:
  S5(int v):a(v) { }
};

S3 h;
#pragma omp threadprivate(h) // expected-note {{defined as threadprivate or thread local}}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  S4 e(4); // expected-note {{'e' defined here}}
  S5 g(5); // expected-note {{'g' defined here}}
  int i;
  int &j = i; // expected-note {{'j' defined here}}
  #pragma omp parallel firstprivate // expected-error {{expected '(' after 'firstprivate'}}
  #pragma omp parallel firstprivate ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel firstprivate () // expected-error {{expected expression}}
  #pragma omp parallel firstprivate (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel firstprivate (argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp parallel firstprivate (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}}
  #pragma omp parallel firstprivate (argc)
  #pragma omp parallel firstprivate (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp parallel firstprivate (a, b, c, d, f) // expected-error {{firstprivate variable with incomplete type 'S1'}}
  #pragma omp parallel firstprivate (argv[1]) // expected-error {{expected variable name}}
  #pragma omp parallel firstprivate(ba)
  #pragma omp parallel firstprivate(ca)
  #pragma omp parallel firstprivate(da)
  #pragma omp parallel firstprivate(S2::S2s)
  #pragma omp parallel firstprivate(S2::S2sc)
  #pragma omp parallel firstprivate(e, g) // expected-error 2 {{firstprivate variable must have an accessible, unambiguous copy constructor}}
  #pragma omp parallel firstprivate(h) // expected-error {{threadprivate or thread local variable cannot be firstprivate}}
  #pragma omp parallel private(i), firstprivate(i) // expected-error {{private variable cannot be firstprivate}} expected-note{{defined as private}}
  foo();
  #pragma omp parallel shared(i)
  #pragma omp parallel firstprivate(i)
  #pragma omp parallel firstprivate(j) // expected-error {{arguments of OpenMP clause 'firstprivate' cannot be of reference type}}
  foo();

  return 0;
}
