// RUN: %clang_cc1 -x c++ -std=c++0x -fsyntax-only -verify %s

// This file should be encoded using ISO-8859-1, the string literals should
// contain the ISO-8859-1 encoding for the code points U+00C0 U+00E9 U+00EE
// U+00F5 U+00FC

void f() {
    wchar_t const *a = L"Àéîõü"; // expected-error {{illegal sequence in string literal}}

    char16_t const *b = u"Àéîõü"; // expected-error {{illegal sequence in string literal}}
    char32_t const *c = U"Àéîõü"; // expected-error {{illegal sequence in string literal}}
    wchar_t const *d = LR"(Àéîõü)"; // expected-error {{illegal sequence in string literal}}
    char16_t const *e = uR"(Àéîõü)"; // expected-error {{illegal sequence in string literal}}
    char32_t const *f = UR"(Àéîõü)"; // expected-error {{illegal sequence in string literal}}
}
