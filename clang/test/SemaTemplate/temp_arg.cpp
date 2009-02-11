// RUN: clang -fsyntax-only -verify %s
template<typename T, 
         int I, 
         template<typename> class TT>
  class A;

template<typename> class X;

A<int, 0, X> * a1;

A<float, 1, X, double> *a2; // expected-error{{too many template arguments for class template 'A'}} \
          // expected-error{{unqualified-id}}
A<float, 1> *a3; // expected-error{{too few template arguments for class template 'A'}} \
          // expected-error{{unqualified-id}}
