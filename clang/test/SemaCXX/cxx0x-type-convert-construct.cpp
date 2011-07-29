// RUN: %clang_cc1 -std=gnu++0x -fsyntax-only -verify %s 

void f() {
  char *u8str;
  u8str = u8"a UTF-8 string"; // expected-error {{assigning to 'char *' from incompatible type 'const char [15]'}}
  char16_t *ustr;
  ustr = u"a UTF-16 string"; // expected-error {{assigning to 'char16_t *' from incompatible type 'const char16_t [16]'}}
  char32_t *Ustr;
  Ustr = U"a UTF-32 string"; // expected-error {{assigning to 'char32_t *' from incompatible type 'const char32_t [16]'}}
}
