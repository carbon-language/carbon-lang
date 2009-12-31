// RUN: %clang_cc1 -fsyntax-only -faccess-control -verify %s
namespace T1 {

class A {
  virtual int f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual void f(); // expected-error{{virtual function 'f' has a different return type ('void') than the function it overrides (which has return type 'int')}}
};

}

namespace T2 {

struct a { };
struct b { };
  
class A {
  virtual a* f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual b* f(); // expected-error{{return type of virtual function 'f' is not covariant with the return type of the function it overrides ('struct T2::b *' is not derived from 'struct T2::a *')}}
};

}

namespace T3 {

struct a { };
struct b : private a { }; // expected-note{{'private' inheritance specifier here}}
  
class A {
  virtual a* f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual b* f(); // expected-error{{return type of virtual function 'f' is not covariant with the return type of the function it overrides (conversion from 'struct T3::b' to inaccessible base class 'struct T3::a')}}
};

}

namespace T4 {

struct a { };
struct a1 : a { };
struct b : a, a1 { };
  
class A {
  virtual a* f(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual b* f(); // expected-error{{return type of virtual function 'f' is not covariant with the return type of the function it overrides (ambiguous conversion from derived class 'struct T4::b' to base class 'struct T4::a':\n\
    struct T4::b -> struct T4::a\n\
    struct T4::b -> struct T4::a1 -> struct T4::a)}}
};

}

namespace T5 {
  
struct a { };

class A {
  virtual a* const f(); 
  virtual a* const g(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual a* const f(); 
  virtual a* g(); // expected-error{{return type of virtual function 'g' is not covariant with the return type of the function it overrides ('struct T5::a *' has different qualifiers than 'struct T5::a *const')}}
};

}

namespace T6 {
  
struct a { };

class A {
  virtual const a* f(); 
  virtual a* g(); // expected-note{{overridden virtual function is here}}
};

class B : A {
  virtual a* f(); 
  virtual const a* g(); // expected-error{{return type of virtual function 'g' is not covariant with the return type of the function it overrides (class type 'struct T6::a const *' is more qualified than class type 'struct T6::a *'}}
};

}

namespace T7 {
  struct a { };
  struct b { };

  class A {
    a* f();
  };

  class B : A {
    virtual b* f();
  };
}

namespace T8 {
  struct a { };
  struct b; // expected-note {{forward declaration of 'struct T8::b'}}
  
  class A {
    virtual a *f();
  };
  
  class B : A {
    b* f(); // expected-error {{return type of virtual function 'f' is not covariant with the return type of the function it overrides ('struct T8::b' is incomplete)}}
  };
}

namespace T9 {
  struct a { };
  
  template<typename T> struct b : a {
    int a[sizeof(T) ? -1 : -1]; // expected-error {{array size is negative}}
  };
  
  class A {
    virtual a *f();
  };
  
  class B : A {
    virtual b<int> *f(); // expected-note {{in instantiation of template class 'struct T9::b<int>' requested here}}
  };
}

// PR5656
class X0 {
  virtual void f0();
};
class X1 : public X0 {
  void f0() = 0;
};

template <typename Base>
struct Foo : Base { 
  void f(int) = 0; // expected-error{{not virtual and cannot be declared pure}}
};

struct Base1 { virtual void f(int); };
struct Base2 { };

void test() {
  (void)sizeof(Foo<Base1>);
  (void)sizeof(Foo<Base2>); // expected-note{{instantiation}}
}

template<typename Base>
struct Foo2 : Base {
  template<typename T> int f(T);
};

void test2() {
  Foo2<Base1> f1;
  Foo2<Base2> f2;
  f1.f(17);
  f2.f(17);
};

struct Foo3 {
  virtual void f(int) = 0; // expected-note{{pure virtual function}}
};

template<typename T>
struct Bar3 : Foo3 {
  void f(T);
};

void test3() {
  Bar3<int> b3i; // okay
  Bar3<float> b3f; // expected-error{{is an abstract class}}
}

// 5920
namespace PR5920 {
  class Base {};

  template <typename T>
  class Derived : public Base {};

  class Foo {
   public:
    virtual Base* Method();
  };

  class Bar : public Foo {
   public:
    virtual Derived<int>* Method();
  };
}
