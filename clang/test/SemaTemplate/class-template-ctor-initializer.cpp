// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class X> struct A {};

template<class X> struct B : A<X> { 
  B() : A<X>() {} 
};
B<int> x;

template<class X> struct B1 : A<X> {
  typedef A<X> Base;
  B1() : Base() {}
};
B1<int> x1;


template<typename T> struct Tmpl { };

template<typename T> struct TmplB { };

struct TmplC : Tmpl<int> {
   TmplC() :
             Tmpl<int>(),
             TmplB<int>() { } // expected-error {{type 'TmplB<int>' is not a direct or virtual base of 'TmplC'}}
};


struct TmplD : Tmpl<char>, TmplB<char> {
    TmplD():
            Tmpl<int>(), // expected-error {{type 'Tmpl<int>' is not a direct or virtual base of 'TmplD'}}
            TmplB<char>() {}
};

namespace PR7259 {
  class Base {
  public:
    Base() {}
  };

  template <class ParentClass>
  class Derived : public ParentClass {
  public:
    Derived() : Base() {}
  };

  class Final : public Derived<Base> {
  };

  int
  main (void)
  {
    Final final;
    return 0;
  }
}
