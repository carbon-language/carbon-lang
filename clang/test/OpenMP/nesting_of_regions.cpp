// RUN: %clang_cc1 -fsyntax-only -fopenmp=libiomp5 -verify %s

void bar();

template <class T>
void foo() {
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i);
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i);
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
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
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
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
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp sections
  {
#pragma omp simd
  for (int i = 0; i < 10; ++i);
  }
#pragma omp sections
  {
#pragma omp parallel
  for (int i = 0; i < 10; ++i);
  }
#pragma omp sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
}

void foo() {
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i);
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i);
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
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
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
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
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
  for (int i = 0; i < 10; ++i);
  }
#pragma omp sections
  {
#pragma omp simd
  for (int i = 0; i < 10; ++i);
  }
#pragma omp sections
  {
#pragma omp parallel
  for (int i = 0; i < 10; ++i);
  }
#pragma omp sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
  return foo<int>();
}

