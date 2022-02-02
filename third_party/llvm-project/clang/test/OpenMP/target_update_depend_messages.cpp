// RUN: %clang_cc1 -verify=expected,omp4 -fopenmp -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify=expected,omp4 -fopenmp-simd -fopenmp-version=45 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
// RUN: %clang_cc1 -verify=expected,omp5 -fopenmp-simd -fopenmp-version=50 -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized
void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note 2 {{declared here}}

class vector {
  public:
    int operator[](int index) { return 0; }
};
template <class T, class S, class R>
int tmain(T argc, S **argv, R *env[]) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;
  char *arr;
  int i, z;

  #pragma omp depend target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp depend(out:argc) target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp target depend(in:argc) update to(z) // expected-error{{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error{{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}

  #pragma omp target update to(z) depend // expected-error {{expected '(' after 'depend'}}
#pragma omp target update to(z) depend(  // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-error {{expected ')'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
#pragma omp target update to(z) depend() // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
#pragma omp target update to(z) depend(argc // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(source : argc) // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
#pragma omp target update to(z) depend(source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}} omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
#pragma omp target update to(z) depend(in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
#pragma omp target update to(z) depend(out:)    // expected-error {{expected expression}}
#pragma omp target update to(z) depend(inout : foobool(argc)), depend(in, argc) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
#pragma omp target update to(z) depend(out : S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target update to(z) depend(in : argv[1][1] = '2')
#pragma omp target update to(z) depend(in : vec[1]) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
#pragma omp target update to(z) depend(in : argv[0])
#pragma omp target update to(z) depend(in:) // expected-error {{expected expression}}
#pragma omp target update to(z) depend(in : tmain)
#pragma omp target update to(z) depend(in : a[0]) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
#pragma omp target update to(z) depend(in : vec [1:2]) // expected-error {{ value is not an array or pointer}}
#pragma omp target update to(z) depend(in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv [argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv [argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv [0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
#pragma omp target update to(z) depend(in : argv [-1:0]) // expected-error {{zero-length array section is not allowed in 'depend' clause}}
#pragma omp target update to(z) depend(in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
#pragma omp target update to(z) depend(in : argv [3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
#pragma omp target update to(z) depend(in : a [0:1]) // expected-error {{subscripted value is not an array or pointer}}
#pragma omp target update to(z) depend(in : argv [argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
#pragma omp target update to(z) depend(in : argv [0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
#pragma omp target update to(z) depend(in : env [0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
#pragma omp target update to(z) depend(in : argv[:argc] [1:argc - 1])
#pragma omp target update to(z) depend(in : arr[0])

  return 0;
}
int main(int argc, char **argv, char *env[]) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;
  auto arr = x; // expected-error {{use of undeclared identifier 'x'}}
  int z;

  #pragma omp depend target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp depend(out:argc) target update to(z) // expected-error{{expected an OpenMP directive}}
  #pragma omp target depend(in:argc) update to(z) // expected-error{{unexpected OpenMP clause 'update' in directive '#pragma omp target'}} expected-error{{unexpected OpenMP clause 'to' in directive '#pragma omp target'}}
  {}

  #pragma omp target update to(z) depend // expected-error {{expected '(' after 'depend'}}
#pragma omp target update to(z) depend(  // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} expected-error {{expected ')'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-note {{to match this '('}} expected-warning {{missing ':' after dependency type - ignoring}}
#pragma omp target update to(z) depend() // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}}
#pragma omp target update to(z) depend(argc // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(source : argc) // omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
#pragma omp target update to(z) depend(source) // expected-error {{expected expression}} expected-warning {{missing ':' after dependency type - ignoring}} omp4-error {{expected 'in', 'out', 'inout' or 'mutexinoutset' in OpenMP clause 'depend'}} omp5-error {{expected depend modifier(iterator) or 'in', 'out', 'inout', 'mutexinoutset' or 'depobj' in OpenMP clause 'depend'}}
#pragma omp target update to(z) depend(in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
#pragma omp target update to(z) depend(out:)    // expected-error {{expected expression}}
#pragma omp target update to(z) depend(inout : foobool(argc)), depend(in, argc) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}} expected-warning {{missing ':' after dependency type - ignoring}} expected-error {{expected expression}}
#pragma omp target update to(z) depend(out : S1) // expected-error {{'S1' does not refer to a value}}
#pragma omp target update to(z) depend(in : argv[1][1] = '2')
#pragma omp target update to(z) depend(in : vec[1]) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
#pragma omp target update to(z) depend(in : argv[0])
#pragma omp target update to(z) depend(in:) // expected-error {{expected expression}}
#pragma omp target update to(z) depend(in : main)
#pragma omp target update to(z) depend(in : a[0]) // omp4-error {{expected addressable lvalue expression, array element or array section}} omp5-error {{expected addressable lvalue expression, array element, array section or array shaping expression of non 'omp_depend_t' type}}
#pragma omp target update to(z) depend(in : vec [1:2]) // expected-error {{ value is not an array or pointer}}
#pragma omp target update to(z) depend(in : argv[ // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv[: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv[:] // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv [argc: // expected-error {{expected expression}} expected-error {{expected ']'}} expected-error {{expected ')'}} expected-note {{to match this '['}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv [argc:argc] // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp target update to(z) depend(in : argv [0:-1]) // expected-error {{section length is evaluated to a negative value -1}}
#pragma omp target update to(z) depend(in : argv [-1:0]) // expected-error {{zero-length array section is not allowed in 'depend' clause}}
#pragma omp target update to(z) depend(in : argv[:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
#pragma omp target update to(z) depend(in : argv [3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
#pragma omp target update to(z) depend(in : a [0:1]) // expected-error {{subscripted value is not an array or pointer}}
#pragma omp target update to(z) depend(in : argv [argv[:2]:1]) // expected-error {{OpenMP array section is not allowed here}}
#pragma omp target update to(z) depend(in : argv [0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is not an array}}
#pragma omp target update to(z) depend(in : env [0:][:]) // expected-error {{section length is unspecified and cannot be inferred because subscripted value is an array of unknown bound}}
#pragma omp target update to(z) depend(in : argv[:argc] [1:argc - 1])
#pragma omp target update to(z) depend(in : arr[0])
  return tmain(argc, argv, env); // expected-note {{in instantiation of function template specialization 'tmain<int, char, char>' requested here}}
}
