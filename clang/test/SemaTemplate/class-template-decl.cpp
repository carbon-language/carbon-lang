// RUN: clang-cc -fsyntax-only -verify %s

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

#if 0
// FIXME: parse template declarations in these scopes, so that we can
// complain about the one at function scope.
class X {
public:
  template<typename T> class C;
};

void f() {
  template<typename T> class X;
}
#endif
