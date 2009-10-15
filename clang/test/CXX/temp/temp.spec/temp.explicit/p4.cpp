// RUN: clang-cc -fsyntax-only -verify %s

template<typename T> void f0(T); // expected-note{{here}}
template void f0(int); // expected-error{{explicit instantiation of undefined function template}}

template<typename T>
struct X0 {
  void f1(); // expected-note{{here}}
  
  static T value; // expected-note{{here}}
};

template void X0<int>::f1(); // expected-error{{explicit instantiation of undefined member function}}

template int X0<int>::value; // expected-error{{explicit instantiation of undefined static data member}}

