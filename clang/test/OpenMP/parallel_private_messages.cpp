// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 %s

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
  static float S2s;
};
const S2 b;
const S2 ba[5];
class S3 {
  int a;
public:
  S3():a(0) { }
};
const S3 c;
const S3 ca[5];
extern const int f;
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

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
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
  #pragma omp parallel private (a, b, c, d, f) // expected-error {{a private variable with incomplete type 'S1'}}
  #pragma omp parallel private (argv[1]) // expected-error {{expected variable name}}
  #pragma omp parallel private(ba)
  #pragma omp parallel private(ca)
  #pragma omp parallel private(da)
  #pragma omp parallel private(S2::S2s)
  #pragma omp parallel private(e, g) // expected-error 2 {{private variable must have an accessible, unambiguous default constructor}}
  foo();
  #pragma omp parallel private(i)
  #pragma omp parallel private(j) // expected-error {{arguments of OpenMP clause 'private' cannot be of reference type 'int &'}}
  foo();

  return 0;
}
