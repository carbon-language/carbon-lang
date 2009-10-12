// RUN: clang-cc -fsyntax-only -verify %s
template<class T1> 
class A {
  template<class T2> class B {
    void mf();
  };
};

template<> template<> class A<int>::B<double>; 
template<> template<> void A<char>::B<char>::mf();

template<> void A<char>::B<int>::mf(); // expected-error{{requires 'template<>'}}
