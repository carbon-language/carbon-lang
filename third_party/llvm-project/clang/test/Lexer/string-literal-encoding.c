// RUN: %clang_cc1 -x c++ -std=c++0x -fsyntax-only -verify %s

// This file should be encoded using ISO-8859-1, the string literals should
// contain the ISO-8859-1 encoding for the code points U+00C0 U+00E9 U+00EE
// U+00F5 U+00FC

void f() {
    wchar_t const *a = L"�����"; // expected-error {{illegal character encoding in string literal}}

    char16_t const *b = u"�����"; // expected-error {{illegal character encoding in string literal}}
    char32_t const *c = U"�����"; // expected-error {{illegal character encoding in string literal}}
    wchar_t const *d = LR"(�����)"; // expected-error {{illegal character encoding in string literal}}
    char16_t const *e = uR"(�����)"; // expected-error {{illegal character encoding in string literal}}
    char32_t const *f = UR"(�����)"; // expected-error {{illegal character encoding in string literal}}

    char const *g = "�����"; // expected-warning {{illegal character encoding in string literal}}
    char const *h = u8"�����"; // expected-error {{illegal character encoding in string literal}}
    char const *i = R"(�����)"; // expected-warning {{illegal character encoding in string literal}}
}

void g() {
    wchar_t const *a = L"foo �����"; // expected-error {{illegal character encoding in string literal}}

    char16_t const *b = u"foo �����"; // expected-error {{illegal character encoding in string literal}}
    char32_t const *c = U"foo �����"; // expected-error {{illegal character encoding in string literal}}
    wchar_t const *d = LR"(foo �����)"; // expected-error {{illegal character encoding in string literal}}
    char16_t const *e = uR"(foo �����)"; // expected-error {{illegal character encoding in string literal}}
    char32_t const *f = UR"(foo �����)"; // expected-error {{illegal character encoding in string literal}}

    char const *g = "foo �����"; // expected-warning {{illegal character encoding in string literal}}
    char const *h = u8"foo �����"; // expected-error {{illegal character encoding in string literal}}
    char const *i = R"(foo �����)"; // expected-warning {{illegal character encoding in string literal}}
}
