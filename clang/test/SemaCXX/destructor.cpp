// RUN: %clang_cc1 -fsyntax-only -verify %s 
class A {
public:
  ~A();
};

class B {
public:
  ~B() { }
};

class C {
public:
  (~C)() { }
};

struct D {
  static void ~D(int, ...) const { } //                          \
    // expected-error{{type qualifier is not allowed on this function}} \
    // expected-error{{destructor cannot be declared 'static'}}  \
    // expected-error{{destructor cannot have any parameters}}   \
    // expected-error{{destructor cannot be variadic}} \
    // expected-error{{destructor cannot have a return type}} \
    // expected-error{{'const' qualifier is not allowed on a destructor}}
};

struct D2 {
  void ~D2() { } //                          \
  // expected-error{{destructor cannot have a return type}}  
};


struct E;

typedef E E_typedef;
struct E {
  ~E_typedef(); // expected-error{{destructor cannot be declared using a typedef 'E_typedef' (aka 'E') of the class name}}
};

struct F {
  (~F)(); // expected-note {{previous declaration is here}}
  ~F(); // expected-error {{destructor cannot be redeclared}}
};

~; // expected-error {{expected a class name after '~' to name a destructor}}
~undef(); // expected-error {{expected the class name after '~' to name a destructor}}
~operator+(int, int);  // expected-error {{expected a class name after '~' to name a destructor}}
~F(){} // expected-error {{destructor must be a non-static member function}}

struct G {
  ~G();
};

G::~G() { }

// <rdar://problem/6841210>
struct H {
  ~H(void) { } 
};

struct X {};

struct Y {
  ~X(); // expected-error {{expected the class name after '~' to name the enclosing class}}
};

namespace PR6421 {
  class T; // expected-note{{forward declaration}}

  class QGenericArgument // expected-note{{declared here}}
  {
    template<typename U>
    void foo(T t) // expected-error{{variable has incomplete type}}
    { }
    
    void disconnect()
    {
      T* t;
      bob<QGenericArgument>(t); // expected-error{{undeclared identifier 'bob'}} \
      // expected-error{{does not refer to a value}}
    }
  };
}

namespace PR6709 {
  template<class T> class X { T v; ~X() { ++*v; } };
  void a(X<int> x) {}
}

struct X0 { virtual ~X0() throw(); };
struct X1 : public X0 { };

// Make sure we instantiate operator deletes when building a virtual
// destructor.
namespace test6 {
  template <class T> class A {
  public:
    void *operator new(__SIZE_TYPE__);
    void operator delete(void *p) {
      T::deleteIt(p); // expected-error {{type 'int' cannot be used prior to '::'}}
    }

    virtual ~A() {}
  };

  class B : A<int> { B(); }; // expected-note {{in instantiation of member function 'test6::A<int>::operator delete' requested here}}
  B::B() {}
}

// Make sure classes are marked invalid when they have invalid
// members.  This avoids a crash-on-invalid.
namespace test7 {
  struct A {
    ~A() const; // expected-error {{'const' qualifier is not allowed on a destructor}}
  };
  struct B : A {};

  void test() {
    B *b;
    b->~B();
  }
}
