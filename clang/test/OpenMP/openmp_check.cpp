// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s
int nested(int a) {
#pragma omp parallel
  ++a;

  auto F = [&]() { // expected-error {{expected expression}} expected-error {{expected ';' at end of declaration}} expected-warning {{'auto' type specifier is a C++11 extension}}
#pragma omp parallel
    {
#pragma omp target
      ++a;
    }
  };
  F(); // expected-error {{C++ requires a type specifier for all declarations}}
  return a; // expected-error {{expected unqualified-id}}
}// expected-error {{extraneous closing brace ('}')}}
