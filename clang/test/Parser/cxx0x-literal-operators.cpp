// RUN: clang-cc -fsyntax-only -verify -std=c++0x %s

void operator "" (); // expected-error {{expected identifier}}
void operator "k" foo(); // expected-error {{string literal after 'operator' must be '""'}} \
                         // expected-error {{C++0x literal operator support is currently under development}}