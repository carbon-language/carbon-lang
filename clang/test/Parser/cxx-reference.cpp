// RUN: clang-cc -fsyntax-only -verify %s

extern char *bork;
char *& bar = bork;

int val;

void foo(int &a) {
}

typedef int & A;

void g(const A aref) {
}

int & const X = val; // expected-error {{'const' qualifier may not be applied to a reference}}
int & volatile Y = val; // expected-error {{'volatile' qualifier may not be applied to a reference}}
int & const volatile Z = val; /* expected-error {{'const' qualifier may not be applied}} \
                           expected-error {{'volatile' qualifier may not be applied}} */

typedef int && RV; // expected-error {{rvalue references are only allowed in C++0x}}
