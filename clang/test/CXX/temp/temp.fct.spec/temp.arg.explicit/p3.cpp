// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class X, class Y, class Z> X f(Y,Z); // expected-note {{candidate function}}

void g() {
  f<int,char*,double>("aa",3.0); 
  f<int,char*>("aa",3.0); // Z is deduced to be double 
  f<int>("aa",3.0);       // Y is deduced to be char*, and
                          // Z is deduced to be double 
  f("aa",3.0); // expected-error{{no matching}}
}

// PR5910
namespace PR5910 {
  template <typename T>
  void Func() {}
  
  template <typename R>
  void Foo(R (*fp)());
  
  void Test() {
    Foo(Func<int>);
  }
}

// PR5949
namespace PR5949 {
  struct Bar;

  template <class Container>
  void quuz(const Container &cont) {
  }

  template<typename T>
  int Foo(Bar *b, void (*Baz)(const T &t), T * = 0) {
    return 0;
  }

  template<typename T>
  int Quux(Bar *b, T * = 0)
  {
    return Foo<T>(b, quuz);
  }
}
