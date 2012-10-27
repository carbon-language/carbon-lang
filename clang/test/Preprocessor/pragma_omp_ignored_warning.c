// RUN: %clang_cc1 -fsyntax-only -verify %s

int pragma_omp_ignored_warning_test() {
  int i, VarA;
  #pragma omp parallel // expected-warning {{pragma omp ignored; did you forget to add '-fopenmp' flag?}}
  {
    #pragma omp for    
    for(i=0; i<10; i++) {
      VarA = 29;
    }
  }
  return VarA;
}
