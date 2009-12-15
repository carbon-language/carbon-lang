// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s 
wchar_t x;

void f(wchar_t p) {
  wchar_t x;
  unsigned wchar_t y; // expected-warning {{'wchar_t' cannot be signed or unsigned}}
  signed wchar_t z; // expected-warning {{'wchar_t' cannot be signed or unsigned}}
  ++x;
}

// PR4502
wchar_t const c = L'c';
int a[c == L'c' ? 1 : -1];
