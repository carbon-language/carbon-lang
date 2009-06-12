// RUN: clang-cc -fsyntax-only -verify %s

template<typename T> struct vector;

template<typename T, int N, template<typename X> class TT>
struct Test0;

template<typename T = int, // expected-error{{default template argument}}
         int N = 17, // expected-error{{default template argument}}
         template<typename X> class TT = ::vector> // expected-error{{default template argument}}
  struct Test0<T*, N, TT> { };
