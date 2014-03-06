// RUN: %clang_cc1 -fopenmp=libiomp5 -verify -DFOPENMP -o - %s
// RUN: %clang_cc1 -verify -o - %s
// expected-no-diagnostics
#ifdef FOPENMP
// -fopenmp=libiomp5 option is specified
#ifndef _OPENMP
#error "No _OPENMP macro is defined with -fopenmp option"
#elsif _OPENMP != 201307
#error "_OPENMP has incorrect value"
#endif //_OPENMP
#else
// No -fopenmp=libiomp5 option is specified
#ifdef _OPENMP
#error "_OPENMP macro is defined without -fopenmp option"
#endif // _OPENMP
#endif // FOPENMP

// RUN: %clang_cc1 -fopenmp=libiomp5 -verify -DFOPENMP -o - %s
// RUN: %clang_cc1 -verify -o - %s
// expected-no-diagnostics
#ifdef FOPENMP
// -fopenmp=libiomp5 option is specified
#ifndef _OPENMP
#error "No _OPENMP macro is defined with -fopenmp option"
#elsif _OPENMP != 201307
#error "_OPENMP has incorrect value"
#endif // _OPENMP
#else
// No -fopenmp=libiomp5 option is specified
#ifdef _OPENMP
#error "_OPENMP macro is defined without -fopenmp option"
#endif // _OPENMP
#endif // FOPENMP
