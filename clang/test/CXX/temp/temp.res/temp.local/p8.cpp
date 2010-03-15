// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace N { 
  enum { C };
  template<class T> class B {
    void f(T);
  }; 
}

template<class C> void N::B<C>::f(C) {
  C b;
}

namespace N {
  enum { D };
  namespace M {
    enum { C , D };
    template<typename C> class X {
      template<typename U> void f(C, U);

      template<typename D> void g(C, D) {
        C c;
        D d;
      }
    };
  }
}

template<typename C> 
template<typename D>
void N::M::X<C>::f(C, D) {
  C c;
  D d;
}
