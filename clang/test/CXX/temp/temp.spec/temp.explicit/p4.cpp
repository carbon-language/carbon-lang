// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> void f0(T); // expected-note{{here}}
template void f0(int); // expected-error{{explicit instantiation of undefined function template}}

template<typename T>
struct X0 {
  struct Inner;
  
  void f1(); // expected-note{{here}}
  
  static T value; // expected-note{{here}}
};

template void X0<int>::f1(); // expected-error{{explicit instantiation of undefined member function}}

template int X0<int>::value; // expected-error{{explicit instantiation of undefined static data member}}

template<> void f0(long);
template void f0(long); // okay

template<> void X0<long>::f1();
template void X0<long>::f1();

template<> struct X0<long>::Inner;
template struct X0<long>::Inner;

template<> long X0<long>::value;
template long X0<long>::value;

template<> struct X0<double>;
template struct X0<double>;

// PR 6458
namespace test0 {
  template <class T> class foo {
    int compare(T x, T y);
  };

  template <> int foo<char>::compare(char x, char y);
  template <class T> int foo<T>::compare(T x, T y) {
    // invalid at T=char; if we get a diagnostic here, we're
    // inappropriately instantiating this template.
    void *ptr = x;
  }
  extern template class foo<char>;
  template class foo<char>;
}
