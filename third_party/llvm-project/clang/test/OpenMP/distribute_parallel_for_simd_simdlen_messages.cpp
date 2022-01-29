// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -std=c++98 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp -std=c++11 %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++98 %s -Wuninitialized

// expected-note@* 0+{{declared here}}

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1;

template <class T, typename S, int N, int ST>
T tmain(T argc, S **argv) {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen // expected-error {{expected '(' after 'simdlen'}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen () // expected-error {{expected expression}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+4 {{expected ')'}} expected-note@+4 {{to match this '('}}
// expected-error@+3 2 {{integral constant expression}} expected-note@+3 0+{{constant expression}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (argc 
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

// expected-error@+3 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (ST // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (1)) // expected-warning {{extra tokens at the end of '#pragma omp distribute parallel for simd' are ignored}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen ((ST > 0) ? 1 + ST : 2)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
// expected-error@+3 2 {{directive '#pragma omp distribute parallel for simd' cannot contain more than one 'simdlen' clause}}
// expected-error@+2 {{argument to 'simdlen' clause must be a strictly positive integer value}}
// expected-error@+1 2 {{integral constant expression}} expected-note@+1 0+{{constant expression}}
#pragma omp distribute parallel for simd simdlen (foobool(argc)), simdlen (true), simdlen (-5)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (S) // expected-error {{'S' does not refer to a value}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#if __cplusplus <= 199711L
  // expected-error@+6 2 {{integral constant expression}} expected-note@+6 0+{{constant expression}}
#else
  // expected-error@+4 2 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (4)
  for (int i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (N) // expected-error {{argument to 'simdlen' clause must be a strictly positive integer value}}
  for (T i = ST; i < N; i++)
    argv[0][i] = argv[0][i] - argv[0][i-ST];

  return argc;
}

int main(int argc, char **argv) {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen // expected-error {{expected '(' after 'simdlen'}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen () // expected-error {{expected expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (4 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (2+2)) // expected-warning {{extra tokens at the end of '#pragma omp distribute parallel for simd' are ignored}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (foobool(1) > 0 ? 1 : 2) // expected-error {{integral constant expression}} expected-note 0+{{constant expression}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
// expected-error@+3 {{integral constant expression}} expected-note@+3 0+{{constant expression}}
// expected-error@+2 2 {{directive '#pragma omp distribute parallel for simd' cannot contain more than one 'simdlen' clause}}
// expected-error@+1 {{argument to 'simdlen' clause must be a strictly positive integer value}}
#pragma omp distribute parallel for simd simdlen (foobool(argc)), simdlen (true), simdlen (-5)
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#if __cplusplus <= 199711L
  // expected-error@+6 {{integral constant expression}} expected-note@+6 0+{{constant expression}}
#else
  // expected-error@+4 {{integral constant expression must have integral or unscoped enumeration type, not 'char *'}}
#endif
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 4; i < 12; i++)
    argv[0][i] = argv[0][i] - argv[0][i-4];

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd simdlen(simdlen(tmain<int, char, -1, -2>(argc, argv) // expected-error 2 {{expected ')'}} expected-note 2 {{to match this '('}} expected-note {{in instantiation of function template specialization 'tmain<int, char, -1, -2>' requested here}}
  foo(); // expected-error {{statement after '#pragma omp distribute parallel for simd' must be a for loop}}

  // expected-note@+1 {{in instantiation of function template specialization 'tmain<int, char, 12, 4>' requested here}}
  return tmain<int, char, 12, 4>(argc, argv);
}

