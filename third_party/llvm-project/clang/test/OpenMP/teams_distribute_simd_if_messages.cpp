// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 %s -Wuninitialized

void foo() {
}

bool foobool(int argc) {
  return argc;
}

void xxx(int argc) {
  int cond; // expected-note {{initialize the variable 'cond' to silence this warning}}
#pragma omp teams distribute simd if(cond) // expected-warning {{variable 'cond' is uninitialized when used here}}
  for (int i = 0; i < 10; ++i)
    ;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  T z;
  int i;
  #pragma omp teams distribute simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (S) // expected-error {{'S' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(argc + z)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc)
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(target : argc) // expected-error {{directive name modifier 'target' is not allowed for '#pragma omp teams distribute simd'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc) if (simd :argc) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'if' clause with 'simd' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

  return 0;
}

int main(int argc, char **argv) {
  int i, z;
  #pragma omp teams distribute simd if // expected-error {{expected '(' after 'if'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if () // expected-error {{expected expression}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp teams distribute simd' are ignored}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc > 0 ? argv[1] : argv[2])
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'if' clause}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (S1) // expected-error {{'S1' does not refer to a value}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc + z) if (for:argc) // expected-error {{directive name modifier 'for' is not allowed for '#pragma omp teams distribute simd'}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc) if (simd :argc) // expected-error {{directive '#pragma omp teams distribute simd' cannot contain more than one 'if' clause with 'simd' name modifier}}
  for (i = 0; i < argc; ++i) foo();
  #pragma omp teams distribute simd if(simd : argc) if (argc) // expected-note {{previous clause with directive name modifier specified here}} expected-error {{no more 'if' clause is allowed}}
  for (i = 0; i < argc; ++i) foo();

  return tmain(argc, argv);
}
