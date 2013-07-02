// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct X0 {
  typedef T* type;
  
  void f0(T);
  void f1(type);
};

template<> void X0<char>::f0(char);
template<> void X0<char>::f1(type);

namespace PR6161 {
  template<typename _CharT>
  class numpunct : public locale::facet // expected-error{{use of undeclared identifier 'locale'}} \
              // expected-error{{expected class name}}
  {
    static locale::id id; // expected-error{{use of undeclared identifier}}
  };
  numpunct<char>::~numpunct(); // expected-error{{expected the class name after '~' to name a destructor}}
}

namespace PR12331 {
  template<typename T> struct S {
    struct U { static const int n = 5; };
    enum E { e = U::n }; // expected-note {{implicit instantiation first required here}}
    int arr[e];
  };
  template<> struct S<int>::U { static const int n = sizeof(int); }; // expected-error {{explicit specialization of 'U' after instantiation}}
}
