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
              // expected-error{{expected class name}} \
  // expected-note{{attempt to specialize declaration here}}
  {
    static locale::id id; // expected-error{{use of undeclared identifier}}
  };
  numpunct<char>::~numpunct(); // expected-error{{template specialization requires 'template<>'}} \
  // expected-error{{specialization of member 'PR6161::numpunct<char>::~numpunct' does not specialize an instantiated member}}
}
