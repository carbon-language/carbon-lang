// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> class A;

extern "C++" {
  template<typename T> class B;
}

namespace N {
  template<typename T> class C;
}

extern "C" {
  template<typename T> class D; // expected-error{{templates must have C++ linkage}}
}

template<class U> class A; // expected-note{{previous template declaration is here}}

template<int N> class A; // expected-error{{template parameter has a different kind in template redeclaration}}

template<int N> class NonTypeTemplateParm;

typedef int INT;

template<INT M> class NonTypeTemplateParm; // expected-note{{previous non-type template parameter with type 'INT' (aka 'int') is here}}

template<long> class NonTypeTemplateParm; // expected-error{{template non-type parameter has a different type 'long' in template redeclaration}}

template<template<typename T> class X> class TemplateTemplateParm;

template<template<class> class Y> class TemplateTemplateParm; // expected-note{{previous template declaration is here}} \
      // expected-note{{previous template template parameter is here}}

template<typename> class TemplateTemplateParm; // expected-error{{template parameter has a different kind in template redeclaration}}

template<template<typename T, int> class X> class TemplateTemplateParm; // expected-error{{too many template parameters in template template parameter redeclaration}}

template<typename T>
struct test {}; // expected-note{{previous definition}}

template<typename T>
struct test : T {}; // expected-error{{redefinition}}

class X {
public:
  template<typename T> class C;
};

void f() {
  template<typename T> class X; // expected-error{{expression}}
}

template<typename T> class X1 { } var; // expected-error{{declared as a template}}

namespace M {
}

template<typename T> class M::C3 { }; // expected-error{{out-of-line definition of 'C3' does not match any declaration in namespace 'M'}}
