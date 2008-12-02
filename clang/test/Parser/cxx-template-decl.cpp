// RUN: clang -fsyntax-only -verify %s

// Errors
export class foo { };   // expected-error {{expected template}}
template  x;            // expected-error {{expected '<' after 'template'}}
export template x;      // expected-error {{expected '<' after 'template'}}
template < ;            // expected-error {{parse error}}
template <template X> ; // expected-error {{expected '<' after 'template'}}
template <template <typename> > ;       // expected-error {{expected 'class' before '>'}}
template <template <typename> Foo> ;    // expected-error {{expected 'class' before 'Foo'}}

// Template function declarations
template <typename T> void foo();
template <typename T, typename U> void foo();

// Template function definitions.
template <typename T> void foo() { }

// Template class (forward) declarations
template <typename T> struct A;
template <typename T, typename U> struct b;
template <typename> struct C;
template <typename, typename> struct D;

// Forward declarations with default parameters?
template <typename T = int> X1;
template <typename = int> X2;

// Forward declarations w/template template parameters
template <template <typename> class T> class TTP1;
template <template <typename> class> class TTP2;
template <template <typename> class T = foo> TTP3;
template <template <typename> class = foo> TTP3;
template <template <typename X, typename Y> class T> TTP5;

// Forward declararations with non-type params
template <int> class NTP0;
template <int N> class NTP1;
template <int N = 5> class NTP2;
template <int = 10> class NTP3;
template <unsigned int N = 12u> NTP4;;
template <unsigned int = 12u> NTP5;
template <unsigned = 15u> NTP6;
template <typename T, T Obj> NTP7;      // expected-error {{parse error}}

// Template class declarations
template <typename T> struct A { };
template <typename T, typename U> struct B { };

