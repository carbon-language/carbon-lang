// RUN: %clang_cc1 -fsyntax-only -fopenmp=libiomp5 -verify %s

template <class T>
void foo() {
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i);
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i);
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
  for (int i = 0; i < 10; ++i);
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
  for (int i = 0; i < 10; ++i);
  }
}

void foo() {
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i);
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i);
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
  for (int i = 0; i < 10; ++i);
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
  for (int i = 0; i < 10; ++i);
  }
  return foo<int>();
}

