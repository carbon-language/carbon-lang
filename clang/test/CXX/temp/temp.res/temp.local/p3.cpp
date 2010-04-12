// RUN: %clang_cc1 -verify %s

template <class T> struct Base { // expected-note 4 {{member found by ambiguous name lookup}}
  static void f();
}; 

struct X0 { };

template <class T> struct Derived: Base<int>, Base<char> {
  typename Derived::Base b;	// expected-error{{member 'Base' found in multiple base classes of different types}}
  typename Derived::Base<double> d;	// OK

  void g(X0 *t) {
    t->Derived::Base<T>::f();
    t->Base<T>::f();
    t->Base::f(); // expected-error{{member 'Base' found in multiple base classes of different types}} \
    // expected-error{{no member named 'f' in 'X0'}} \
    // expected-error{{expected a class or namespace}}
  }
};

namespace PR6717 {
  template <typename T>
  class WebVector {
  }

    WebVector(const WebVector<T>& other) { } 

  template <typename C>
  WebVector<T>& operator=(const C& other) { } // expected-error{{unknown type name 'WebVector'}} \
  // expected-error{{unqualified-id}}
}
