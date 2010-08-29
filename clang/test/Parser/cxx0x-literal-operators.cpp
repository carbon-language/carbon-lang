// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

void operator "" (const char *); // expected-error {{expected identifier}}
void operator "k"_foo(const char *); // expected-error {{string literal after 'operator' must be '""'}}
void operator ""_tester (const char *);
