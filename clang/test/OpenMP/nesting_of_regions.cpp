// RUN: %clang_cc1 -fsyntax-only -fopenmp=libiomp5 -verify %s

void bar();

template <class T>
void foo() {
// PARALLEL DIRECTIVE
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel region}}
  {
    bar();
  }
#pragma omp parallel
#pragma omp single
  bar();
#pragma omp parallel
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp parallel sections
  {
    bar();
  }

// SIMD DIRECTIVE
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

// FOR DIRECTIVE
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a for region}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }

// SECTIONS DIRECTIVE
#pragma omp sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }

// SECTION DIRECTIVE
#pragma omp section // expected-error {{orphaned 'omp section' directives are prohibited, it must be closely nested to a sections region}}
  {
    bar();
  }

// SINGLE DIRECTIVE
#pragma omp single
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp single
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel sections
    {
      bar();
    }
  }

// PARALLEL FOR DIRECTIVE
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel for region}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }

// PARALLEL SECTIONS DIRECTIVE
#pragma omp parallel sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
}

void foo() {
// PARALLEL DIRECTIVE
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel region}}
  {
    bar();
  }
#pragma omp parallel
#pragma omp sections
  {
    bar();
  }
#pragma omp parallel
#pragma omp single
  bar();
#pragma omp parallel
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp parallel sections
  {
    bar();
  }

// SIMD DIRECTIVE
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp simd // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    {
      bar();
    }
  }

// FOR DIRECTIVE
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a for region}}
    {
      bar();
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    bar();
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }

// SECTIONS DIRECTIVE
#pragma omp sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp sections
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    bar();
  }
#pragma omp sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }

// SECTION DIRECTIVE
#pragma omp section // expected-error {{orphaned 'omp section' directives are prohibited, it must be closely nested to a sections region}}
  {
    bar();
  }

// SINGLE DIRECTIVE
#pragma omp single
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp single // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'single' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp single
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp single
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp single
  {
#pragma omp parallel sections
    {
      bar();
    }
  }

// PARALLEL FOR DIRECTIVE
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp section // expected-error {{'omp section' directive must be closely nested to a sections region, not a parallel for region}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp single // expected-error {{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp parallel sections
    {
      bar();
    }
  }

// PARALLEL SECTIONS DIRECTIVE
#pragma omp parallel sections
  {
#pragma omp for // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp simd
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp sections // expected-error {{region cannot be closely nested inside 'parallel sections' region; perhaps you forget to enclose 'omp sections' directive into a parallel region?}}
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp single // expected-error {{region cannot be closely nested inside 'section' region; perhaps you forget to enclose 'omp single' directive into a parallel region?}}
      bar();
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel
    {
#pragma omp single // OK
      {
        bar();
      }
#pragma omp for // OK
      for (int i = 0; i < 10; ++i)
        ;
#pragma omp sections // OK
      {
        bar();
      }
    }
  }
#pragma omp parallel sections
  {
#pragma omp parallel for
    for (int i = 0; i < 10; ++i)
      ;
  }
#pragma omp parallel sections
  {
#pragma omp parallel sections
    {
      bar();
    }
  }
  return foo<int>();
}

