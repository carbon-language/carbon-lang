// RUN: clang-cc -fsyntax-only -verify %s
class X; // expected-note{{here}}
typedef struct X * X_t;

template<typename T> class Y;
template<class U> struct Y { };

union X { int x; float y; }; // expected-error{{use of 'X' with tag type that does not match previous declaration}}
