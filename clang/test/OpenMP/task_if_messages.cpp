// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

template <class T, class S> // expected-note {{declared here}}
int tmain(T argc, S **argv) {
  #pragma omp task if // expected-error {{expected '(' after 'if'}}
  #pragma omp task if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if () // expected-error {{expected expression}}
  #pragma omp task if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task if (argc > 0 ? argv[1] : argv[2])
  #pragma omp task if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp task' cannot contain more than one 'if' clause}}
  #pragma omp task if (S) // expected-error {{'S' does not refer to a value}}
  #pragma omp task if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if(argc)
  foo();

  return 0;
}

int main(int argc, char **argv) {
  #pragma omp task if // expected-error {{expected '(' after 'if'}}
  #pragma omp task if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if () // expected-error {{expected expression}}
  #pragma omp task if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task if (argc > 0 ? argv[1] : argv[2])
  #pragma omp task if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp task' cannot contain more than one 'if' clause}}
  #pragma omp task if (S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp task if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if (argc argc) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if (1 0) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task if(if(tmain(argc, argv) // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();

  return tmain(argc, argv);
}
