// RUN: clang-cc -fsyntax-only -std=c++98 -verify %s
template<int N> struct A; // expected-note 5{{template parameter is declared here}}

A<0> *a0;

A<int()> *a1; // expected-error{{template argument for non-type template parameter is treated as type 'int ()'}}

A<int> *a2; // expected-error{{template argument for non-type template parameter must be an expression}}

A<1 >> 2> *a3; // expected-warning{{use of right-shift operator ('>>') in template argument will require parentheses in C++0x}}

// C++ [temp.arg.nontype]p5:
A<A> *a4; // expected-error{{must have an integral or enumeration type}} \
          // FIXME: the error message above is a bit lame

enum E { Enumerator = 17 };
A<E> *a5; // expected-error{{template argument for non-type template parameter must be an expression}}
template<E Value> struct A1; // expected-note{{template parameter is declared here}}
A1<Enumerator> *a6; // okay
A1<17> *a7; // expected-error{{non-type template argument of type 'int' cannot be converted to a value of type 'enum E'}}

const long LongValue = 12345678;
A<LongValue> *a8;
const short ShortValue = 17;
A<ShortValue> *a9;

int f(int);
A<f(17)> *a10; // expected-error{{non-type template argument of type 'int' is not an integral constant expression}}

class X {
public:
  X();
  X(int, int);
  operator int() const;
};
A<X(17, 42)> *a11; // expected-error{{non-type template argument of type 'class X' must have an integral or enumeration type}}

template<X const *Ptr> struct A2;

X *X_ptr;
X an_X;
X array_of_Xs[10];
A2<X_ptr> *a12;
A2<array_of_Xs> *a13;
A2<&an_X> *a13_2;
A2<(&an_X)> *a13_3; // expected-error{{non-type template argument cannot be surrounded by parentheses}}

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
A3<h2> *a14_6;  // expected-error{{non-type template argument of type 'float (*)(float)' cannot be converted to a value of type 'int (*)(int)'}}
A3<g> *a14_7; // expected-error{{non-type template argument of type '<overloaded function type>' cannot be converted to a value of type 'int (*)(int)'}}
// FIXME: the first error includes the string <overloaded function
// type>, which makes Doug slightly unhappy.


struct Y { } y;

volatile X * X_volatile_ptr;
template<X const &AnX> struct A4; // expected-note 2{{template parameter is declared here}}
A4<an_X> *a15_1; // okay
A4<*X_volatile_ptr> *a15_2; // expected-error{{reference binding of non-type template parameter of type 'class X const &' to template argument of type 'class X volatile' ignores qualifiers}}
A4<y> *15_3; //  expected-error{{non-type template parameter of reference type 'class X const &' cannot bind to template argument of type 'struct Y'}} \
            // FIXME: expected-error{{expected unqualified-id}}

template<int (&fr)(int)> struct A5; // expected-note 2{{template parameter is declared here}}
A5<h> *a16_1;
A5<f> *a16_3;
A5<h2> *a16_6;  // expected-error{{non-type template argument of type 'float (float)' cannot be converted to a value of type 'int (&)(int)'}}
A5<g> *a14_7; // expected-error{{non-type template argument of type '<overloaded function type>' cannot be converted to a value of type 'int (&)(int)'}}
// FIXME: the first error includes the string <overloaded function
// type>, which makes Doug slightly unhappy.

struct Z {
  int foo(int);
  float bar(float);
  int bar(int);
  double baz(double);

  int int_member;
  float float_member;
};
template<int (Z::*pmf)(int)> struct A6; // expected-note{{template parameter is declared here}}
A6<&Z::foo> *a17_1;
A6<&Z::bar> *a17_2;
A6<&Z::baz> *a17_3; // expected-error{{non-type template argument of type 'double (struct Z::*)(double)' cannot be converted to a value of type 'int (struct Z::*)(int)'}}


template<int Z::*pm> struct A7;  // expected-note{{template parameter is declared here}}
template<int Z::*pm> struct A7c;
A7<&Z::int_member> *a18_1;
A7c<&Z::int_member> *a18_2;
A7<&Z::float_member> *a18_3; // expected-error{{non-type template argument of type 'float struct Z::*' cannot be converted to a value of type 'int struct Z::*'}}
A7c<(&Z::int_member)> *a18_3; // expected-error{{non-type template argument cannot be surrounded by parentheses}}

template<unsigned char C> struct Overflow; // expected-note{{template parameter is declared here}}

Overflow<5> *overflow1; // okay
Overflow<256> *overflow2; // expected-error{{non-type template argument value '256' is too large for template parameter of type 'unsigned char'}}


template<unsigned> struct Signedness; // expected-note{{template parameter is declared here}}
Signedness<10> *signedness1; // okay
Signedness<-10> *signedness2; // expected-error{{non-type template argument provides negative value '-10' for unsigned template parameter of type 'unsigned int'}}

// Check canonicalization of template arguments.
template<int (*)(int, int)> struct FuncPtr0;
int func0(int, int);
extern FuncPtr0<&func0> *fp0;
template<int (*)(int, int)> struct FuncPtr0;
extern FuncPtr0<&func0> *fp0;
int func0(int, int);
extern FuncPtr0<&func0> *fp0;

// PR5350
namespace ns {
  template <typename T>
  struct Foo {
    static const bool value = true;
  };
  
  template <bool b>
  struct Bar {};
  
  const bool value = false;
  
  Bar<bool(ns::Foo<int>::value)> x;
}

// PR5349
namespace ns {
  enum E { k };
  
  template <E e>
  struct Baz  {};
  
  Baz<k> f1;  // This works.
  Baz<E(0)> f2;  // This too.
  Baz<static_cast<E>(0)> f3;  // And this.
  
  Baz<ns::E(0)> b1;  // This doesn't work.
  Baz<static_cast<ns::E>(0)> b2;  // This neither.  
}

