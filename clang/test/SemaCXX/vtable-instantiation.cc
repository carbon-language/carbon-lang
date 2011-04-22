// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR8640

template<class T1> struct C1 {
  virtual void c1() {
    T1 t1 = 3;  // expected-error {{cannot initialize a variable}}
  }
};

template<class T2> struct C2 {
  void c2() {
    new C1<T2>();  // expected-note {{in instantiation of member function}}
  }
};

void f() {
  C2<int*> c2;
  c2.c2();  // expected-note {{in instantiation of member function}}
}

namespace PR9325 {
  template<typename T>
  class Target
  {
  public:
    virtual T Value() const
    {
      return 1; // expected-error{{cannot initialize return object of type 'int *' with an rvalue of type 'int'}}
    }
  };

  template<typename T>
  struct Provider
  {
    static Target<T> Instance;
  };

  template<typename T>
  Target<T> Provider<T>::Instance; // expected-note{{in instantiation of}}

  void f()
  {
    Target<int*>* traits = &Provider<int*>::Instance;
  }

}
