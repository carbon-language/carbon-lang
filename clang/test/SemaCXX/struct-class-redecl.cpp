// RUN: clang-cc -fsyntax-only -Wmismatched-tags -verify %s
class X; // expected-note 2{{here}}
typedef struct X * X_t; // expected-warning{{previously declared}}

template<typename T> struct Y; // expected-note{{previous}}
template<class U> class Y { }; // expected-warning{{previously declared}}

union X { int x; float y; }; // expected-error{{use of 'X' with tag type that does not match previous declaration}}
