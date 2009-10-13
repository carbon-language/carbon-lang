// RUN: clang-cc -fsyntax-only -verify %s
template<class T1> class A { 
  template<class T2> class B {
    template<class T3> void mf1(T3); 
    void mf2();
  };
}; 

template<> template<class X>
class A<long>::B { }; 

// FIXME: If we make the explicit specialization of A<long>::B, above, into
// a specialization of A<int>::B, our diagnostic is correct but not very 
// helpful.
template<> template<> template<class T>
  void A<int>::B<double>::mf1(T t) { } 

// FIXME: This diagnostic could probably be better.
template<class Y> template<>
  void A<Y>::B<double>::mf2() { } // expected-error{{does not refer}}
