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


// PR5917
template<typename _CharT>
struct basic_string {
};

template<typename _CharT>
basic_string<_CharT> operator+ (const basic_string<_CharT>&, _CharT);

int t(void) {
  basic_string<wchar_t>() + L'-';
  return (0);
}
