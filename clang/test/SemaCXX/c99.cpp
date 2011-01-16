// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s
void f1(int i[static 5]) { // expected-error{{C99}}
}

struct Point { int x; int y; };

Point p1 = { .x = 17, // expected-warning{{designated initializers are a C99 feature, accepted in C++ as an extension}} 
             y: 25 }; // expected-warning{{designated initializers are a C99 feature, accepted in C++ as an extension}} \
                      // expected-warning{{use of GNU old-style field designator extension}}
