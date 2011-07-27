// RUN: %clang_cc1 -std=gnu++0x -fsyntax-only -verify %s 
// Runs in c++0x mode so that char16_t and char32_t are available.

void f() {
  float v1 = float(1);
  int v2 = typeof(int)(1,2); // expected-error {{excess elements in scalar initializer}}
  typedef int arr[];
  int v3 = arr(); // expected-error {{array types cannot be value-initialized}}
  int v4 = int();
  int v5 = int; // expected-error {{expected '(' for function-style cast or type construction}}
  typedef int T;
  int *p;
  bool v6 = T(0) == p;
  char *str;
  str = "a string"; // expected-warning{{conversion from string literal to 'char *' is deprecated}}
  wchar_t *wstr;
  wstr = L"a wide string"; // expected-warning{{conversion from string literal to 'wchar_t *' is deprecated}}
  char16_t *ustr;
  ustr = u"a UTF-16 string"; // expected-error {{assigning to 'char16_t *' from incompatible type 'const char16_t [16]'}}
  char32_t *Ustr;
  Ustr = U"a UTF-32 string"; // expected-error {{assigning to 'char32_t *' from incompatible type 'const char32_t [16]'}}
}
