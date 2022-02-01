// RUN: %clang_cc1 -verify -fopenmp -std=c++11 -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++11 -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

template <typename T, int C> // expected-note {{declared here}}
T tmain(T argc) {
  char **a;
  T z;
#pragma omp target teams num_teams(C)
  foo();
#pragma omp target teams num_teams(T) // expected-error {{'T' does not refer to a value}}
  foo();
#pragma omp target teams num_teams // expected-error {{expected '(' after 'num_teams'}}
  foo();
#pragma omp target teams num_teams( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams num_teams() // expected-error {{expected expression}}
  foo();
#pragma omp target teams num_teams(argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
#pragma omp target teams num_teams(argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();
#pragma omp target teams num_teams(argc > 0 ? a[1] : a[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();
#pragma omp target teams num_teams(argc + argc + z)
  foo();
#pragma omp target teams num_teams(argc), num_teams (argc+1) // expected-error {{directive '#pragma omp target teams' cannot contain more than one 'num_teams' clause}}
  foo();
#pragma omp target teams num_teams(S1) // expected-error {{'S1' does not refer to a value}}
  foo();
#pragma omp target teams num_teams(-2) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  foo();
#pragma omp target teams num_teams(-10u)
  foo();
#pragma omp target teams num_teams(3.14) // expected-error 2 {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();

  return 0;
}

int main(int argc, char **argv) {
  int z;
#pragma omp target teams num_teams // expected-error {{expected '(' after 'num_teams'}}
  foo();

#pragma omp target teams num_teams ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();

#pragma omp target teams num_teams () // expected-error {{expected expression}}
  foo();

#pragma omp target teams num_teams (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();

#pragma omp target teams num_teams (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams' are ignored}}
  foo();

#pragma omp target teams num_teams (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}}
  foo();

#pragma omp target teams num_teams (argc + argc*z)
  foo();

#pragma omp target teams num_teams (argc), num_teams (argc+1) // expected-error {{directive '#pragma omp target teams' cannot contain more than one 'num_teams' clause}}
  foo();

#pragma omp target teams num_teams (S1) // expected-error {{'S1' does not refer to a value}}
  foo();

#pragma omp target teams num_teams (-2) // expected-error {{argument to 'num_teams' clause must be a strictly positive integer value}}
  foo();

#pragma omp target teams num_teams (-10u)
  foo();

#pragma omp target teams num_teams (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}
  foo();

  return tmain<int, 10>(argc); // expected-note {{in instantiation of function template specialization 'tmain<int, 10>' requested here}}
}
