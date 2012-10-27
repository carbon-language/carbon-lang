// RUN: %clang_cc1 -E -verify %s
// RUN: %clang_cc1 -E -verify -fopenmp %s
// RUN: %clang_cc1 -Eonly -verify %s
// RUN: %clang_cc1 -Eonly -verify -fopenmp %s
// RUN: %clang_cc1 -E -P -verify %s
// RUN: %clang_cc1 -E -P -verify -fopenmp %s

int pragma_omp_test() {
  int i, VarA;
  #pragma omp parallel // expected-no-diagnostics 
  {
    #pragma omp for    // expected-no-diagnostics 
    for(i=0; i<10; i++) {
      VarA = 29;
    }
  }
  return VarA;
}
