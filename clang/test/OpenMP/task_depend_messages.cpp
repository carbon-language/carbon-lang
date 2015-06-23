// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

class vector {
  public:
    int operator[](int index) { return 0; }
};

int main(int argc, char **argv) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;

  #pragma omp task depend // expected-error {{expected '(' after 'depend'}}
  #pragma omp task depend ( // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp task depend () // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
  #pragma omp task depend (argc // expected-error {{expected 'in', 'out' or 'inout' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp task depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  #pragma omp task depend (out: ) // expected-error {{expected expression}}
  #pragma omp task depend (inout : foobool(argc)), depend (in, argc) // expected-error {{expected variable name, array element or array section}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
  #pragma omp task depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp task depend(in : argv[1][1] = '2') // expected-error {{expected variable name, array element or array section}}
  #pragma omp task depend (in : vec[1]) // expected-error {{expected variable name, array element or array section}}
  #pragma omp task depend (in : argv[0])
  #pragma omp task depend (in : ) // expected-error {{expected expression}}
  #pragma omp task depend (in : main) // expected-error {{expected variable name, array element or array section}}
  #pragma omp task depend(in : a[0]) // expected-error{{expected variable name, array element or array section}}
  foo();

  return 0;
}
