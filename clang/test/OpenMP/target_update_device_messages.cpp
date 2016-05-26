// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

template <class T, class S>
int tmain(T argc, S **argv) {
  int i;
#pragma omp target update to(i) device // expected-error {{expected '(' after 'device'}}
#pragma omp target update to(i) device ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(i) device () // expected-error {{expected expression}}
#pragma omp target update to(i) device (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(i) device (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
#pragma omp target update device (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update device (argc + argc)  // expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update device (argc), device (argc+1) // expected-error {{directive '#pragma omp target update' cannot contain more than one 'device' clause}}  expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update device (S1) // expected-error {{'S1' does not refer to a value}}  expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update device (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}  expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update device (-2) // expected-error {{argument to 'device' clause must be a non-negative integer value}}  expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
}

int main(int argc, char **argv) {
  int j;
#pragma omp target update to(j) device // expected-error {{expected '(' after 'device'}}
#pragma omp target update device ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(j) device () // expected-error {{expected expression}}
#pragma omp target update device (argc // expected-error {{expected ')'}} expected-note {{to match this '('}}  expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(j) device (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
#pragma omp target update device (argc > 0 ? argv[1] : argv[2]) // expected-error {{expression must have integral or unscoped enumeration type, not 'char *'}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(j) device (argc + argc)
#pragma omp target update device (argc), device (argc+1) // expected-error {{directive '#pragma omp target update' cannot contain more than one 'device' clause}}  expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(j) device (S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target update device (-2) // expected-error {{argument to 'device' clause must be a non-negative integer value}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
#pragma omp target update to(j) device (3.14) // expected-error {{expression must have integral or unscoped enumeration type, not 'double'}}

  return tmain(argc, argv);
}
