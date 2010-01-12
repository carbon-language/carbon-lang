// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct X0 : T::template apply<U> { 
  X0(U u) : T::template apply<U>(u) { }
};

template<typename T, typename U>
struct X1 : T::apply<U> { }; // expected-error{{missing 'template' keyword prior to dependent template name 'T::apply'}}

template<typename T>
struct X2 : vector<T> { }; // expected-error{{unknown template name 'vector'}}
