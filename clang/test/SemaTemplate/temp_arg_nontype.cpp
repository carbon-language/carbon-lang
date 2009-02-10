// RUN: clang -fsyntax-only -std=c++98 -verify %s
template<int N> struct A; // expected-note 5{{template parameter is declared here}}

A<0> *a0;

A<int()> *a1; // expected-error{{template argument for non-type template parameter is treated as type 'int (void)'}}

A<int> *a2; // expected-error{{template argument for non-type template parameter must be an expression}}

A<1 >> 2> *a3;

// C++ [temp.arg.nontype]p5:
A<A> *a4; // expected-error{{must have an integral or enumeration type}} \
          // FIXME: the error message above is a bit lame \
          // FIXME: expected-error{{expected unqualified-id}}

enum E { Enumerator = 17 };
A<E> *a5; // expected-error{{template argument for non-type template parameter must be an expression}}

template<E Value> struct A1; // expected-note{{template parameter is declared here}}
A1<Enumerator> *a6; // okay
A1<17> *a7; // expected-error{{non-type template argument of type 'int' cannot be converted to a value of type 'enum E'}} \
          // FIXME: expected-error{{expected unqualified-id}}

const long LongValue = 12345678;
A<LongValue> *a8;
const short ShortValue = 17;
A<ShortValue> *a9;

int f(int);
A<f(17)> *a10; // expected-error{{non-type template argument of type 'int' is not an integral constant expression}} \
          // FIXME: expected-error{{expected unqualified-id}}

class X {
public:
  X(int, int);
  operator int() const;
};
A<X(17, 42)> *a11; // expected-error{{non-type template argument of type 'class X' must have an integral or enumeration type}} \
                   // FIXME:expected-error{{expected unqualified-id}}
