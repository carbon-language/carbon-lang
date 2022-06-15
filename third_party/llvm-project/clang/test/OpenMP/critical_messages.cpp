// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

int foo();

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp critical
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

template<typename T, int N>
int tmain(int argc, char **argv) { // expected-note {{declared here}}
  #pragma omp critical
  ;
  #pragma omp critical untied allocate(argc) // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp critical'}} expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp critical'}}
  #pragma omp critical unknown // expected-warning {{extra tokens at the end of '#pragma omp critical' are ignored}}
  #pragma omp critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name2 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name1)
  foo();
  {
    #pragma omp critical
  } // expected-error {{expected statement}}
  #pragma omp critical (name2) // expected-note {{previous 'critical' region starts here}}
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      foo();
      #pragma omp critical(name2) // expected-error {{cannot nest 'critical' regions having the same name 'name2'}}
      foo();
    }
  }
  #pragma omp critical (name2)
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp critical
      foo();
    }
  }
  #pragma omp critical (name2)
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp critical (nam)
      foo();
    }
  }

  #pragma omp critical (name2) hint // expected-error {{expected '(' after 'hint'}}
  foo();
  #pragma omp critical (name2) hint( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(+ // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(argc) // expected-error {{integral constant expression}} expected-note 0+{{constant expression}}
  foo();
  #pragma omp critical (name) hint(N) // expected-error {{argument to 'hint' clause must be a non-negative integer value}} expected-error {{constructs with the same name must have a 'hint' clause with the same value}} expected-note {{'hint' clause with value '4'}}
  foo();
  #pragma omp critical hint(N) // expected-error {{the name of the construct must be specified in presence of 'hint' clause}}
  foo();

  const int omp_lock_hint_none = 0;
  #pragma omp critical (name3) hint(omp_lock_hint_none)
  foo();
  return 0;
}

int main(int argc, char **argv) { // expected-note {{declared here}}
  #pragma omp critical
  ;
  #pragma omp critical untied // expected-error {{unexpected OpenMP clause 'untied' in directive '#pragma omp critical'}}
  #pragma omp critical unknown // expected-warning {{extra tokens at the end of '#pragma omp critical' are ignored}}
  #pragma omp critical ( // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical ( + // expected-error {{expected identifier}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name2 // expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp critical (name1)
  foo();
  {
    #pragma omp critical
  } // expected-error {{expected statement}}
  #pragma omp critical (name2) // expected-note {{previous 'critical' region starts here}}
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      foo();
      #pragma omp critical(name2) // expected-error {{cannot nest 'critical' regions having the same name 'name2'}}
      foo();
    }
  }
  #pragma omp critical (name2)
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp critical
      foo();
    }
  }
  #pragma omp critical (name2)
  #pragma omp critical
  for (int i = 0; i < 10; ++i) {
    foo();
    #pragma omp parallel
    #pragma omp for
    for (int j = 0; j < 10; j++) {
      #pragma omp critical (nam)
      foo();
    }
  }

  #pragma omp critical (name2) hint // expected-error {{expected '(' after 'hint'}}
  foo();
  #pragma omp critical (name2) hint( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(+ // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  foo();
  #pragma omp critical (name2) hint(argc) // expected-error {{integral constant expression}} expected-note 0+{{constant expression}}
  foo();
  #pragma omp critical (name) hint(23) // expected-note {{previous 'hint' clause with value '23'}}
  foo();
  #pragma omp critical hint(-5) // expected-error {{argument to 'hint' clause must be a non-negative integer value}}
  foo();
  #pragma omp critical hint(1) // expected-error {{the name of the construct must be specified in presence of 'hint' clause}}
  foo();
  return tmain<int, 4>(argc, argv) + tmain<float, -5>(argc, argv); // expected-note {{in instantiation of function template specialization 'tmain<float, -5>' requested here}} expected-note {{in instantiation of function template specialization 'tmain<int, 4>' requested here}}
}

int foo() {
  L1: // expected-note {{jump exits scope of OpenMP structured block}}
    foo();
  #pragma omp critical
  {
    foo();
    goto L1; // expected-error {{cannot jump from this goto statement to its label}}
  }
  goto L2; // expected-error {{cannot jump from this goto statement to its label}}
  #pragma omp critical
  {  // expected-note {{jump bypasses OpenMP structured block}}
    L2:
    foo();
  }

  return 0;
 }
