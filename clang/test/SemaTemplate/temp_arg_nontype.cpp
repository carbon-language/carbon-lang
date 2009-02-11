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
  X();
  X(int, int);
  operator int() const;
};
A<X(17, 42)> *a11; // expected-error{{non-type template argument of type 'class X' must have an integral or enumeration type}} \
                   // FIXME:expected-error{{expected unqualified-id}}

template<X const *Ptr> struct A2;

X *X_ptr;
X array_of_Xs[10];
A2<X_ptr> *a12;
A2<array_of_Xs> *a13;

float f(float);

float g(float);
double g(double);

int h(int);
float h2(float);

template<int fp(int)> struct A3; // expected-note 2{{template parameter is declared here}}
A3<h> *a14_1;
A3<&h> *a14_2;
A3<f> *a14_3;
A3<&f> *a14_4;
A3<((&f))> *a14_5;
A3<h2> *a14_6;  // expected-error{{non-type template argument of type 'float (*)(float)' cannot be converted to a value of type 'int (*)(int)'}} \
// FIXME: expected-error{{expected unqualified-id}}
A3<g> *a14_7; // expected-error{{non-type template argument of type '<overloaded function type>' cannot be converted to a value of type 'int (*)(int)'}}\
// FIXME: expected-error{{expected unqualified-id}}
// FIXME: the first error includes the string <overloaded function
// type>, which makes Doug slightly unhappy.


struct Y { } y;

volatile X * X_volatile_ptr;
template<X const &AnX> struct A4; // expected-note 2{{template parameter is declared here}}
A4<*X_ptr> *a15_1; // okay
A4<*X_volatile_ptr> *a15_2; // expected-error{{reference binding of non-type template parameter of type 'class X const &' to template argument of type 'class X volatile' ignores qualifiers}} \
                  // FIXME: expected-error{{expected unqualified-id}}
A4<y> *15_3; //  expected-error{{non-type template parameter of reference type 'class X const &' cannot bind to template argument of type 'struct Y'}}\
                  // FIXME: expected-error{{expected unqualified-id}}

template<int (&fr)(int)> struct A5; // expected-note 2{{template parameter is declared here}}
A5<h> *a16_1;
A5<(h)> *a16_2;
A5<f> *a16_3;
A5<(f)> *a16_4;
A5<h2> *a16_6;  // expected-error{{non-type template argument of type 'float (float)' cannot be converted to a value of type 'int (&)(int)'}} \
// FIXME: expected-error{{expected unqualified-id}}
A5<g> *a14_7; // expected-error{{non-type template argument of type '<overloaded function type>' cannot be converted to a value of type 'int (&)(int)'}}\
// FIXME: expected-error{{expected unqualified-id}}
// FIXME: the first error includes the string <overloaded function
// type>, which makes Doug slightly unhappy.

struct Z {
  int foo(int);
  float bar(float);
  int bar(int);
  double baz(double);
};
template<int (Z::*pmf)(int)> struct A6; // expected-note{{template parameter is declared here}}
A6<&Z::foo> *a17_1;
A6<&Z::bar> *a17_2;
A6<&Z::baz> *a17_3; // expected-error{{non-type template argument of type 'double (struct Z::*)(double)' cannot be converted to a value of type 'int (struct Z::*)(int)'}} \
// FIXME: expected-error{{expected unqualified-id}}
