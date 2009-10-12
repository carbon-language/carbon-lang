// RUN: clang-cc -fsyntax-only -verify %s
template<class T1> class A { 
  template<class T2> class B {
    template<class T3> void mf1(T3); 
    void mf2();
  };
}; 

template<> template<class X>
class A<int>::B { }; 

template<> template<> template<class T>
  void A<int>::B<double>::mf1(T t) { } 

template<class Y> template<>
  void A<Y>::B<double>::mf2() { } // expected-error{{does not refer}}
