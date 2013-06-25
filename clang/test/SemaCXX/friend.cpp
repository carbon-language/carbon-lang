// RUN: %clang_cc1 -fsyntax-only -verify %s

friend class A; // expected-error {{'friend' used outside of class}}
void f() { friend class A; } // expected-error {{'friend' used outside of class}}
class C { friend class A; };
class D { void f() { friend class A; } }; // expected-error {{'friend' used outside of class}}

// PR5760
namespace test0 {
  namespace ns {
    void f(int);
  }

  struct A {
    friend void ns::f(int a);
  };
}

// Test derived from LLVM's Registry.h
namespace test1 {
  template <class T> struct Outer {
    void foo(T);
    struct Inner {
      friend void Outer::foo(T);
    };
  };

  void test() {
    (void) Outer<int>::Inner();
  }
}

// PR5476
namespace test2 {
  namespace foo {
    void Func(int x);
  }

  class Bar {
    friend void ::test2::foo::Func(int x);
  };
}

// PR5134
namespace test3 {
  class Foo {
    friend const int getInt(int inInt = 0) {}

  };
}

namespace test4 {
  class T4A {
    friend class T4B;
  
  public:
    T4A(class T4B *);

  protected:
    T4B *mB;          // error here
  };
 
  class T4B {};
}

namespace rdar8529993 {
struct A { ~A(); };

struct B : A
{
  template<int> friend A::~A(); // expected-error {{destructor cannot be declared as a template}}
};
}

// PR7915
namespace test5 {
  struct A;
  struct A1 { friend void A(); };

  struct B { friend void B(); };
}

// PR8479
namespace test6_1 {
  class A {
   public:
   private:
    friend class vectorA;
    A() {}
  };
  class vectorA {
   public:
    vectorA(int i, const A& t = A()) {}
  };
  void f() {
    vectorA v(1);
  }
}
namespace test6_2 {
  template<class T>
  class vector {
   public:
    vector(int i, const T& t = T()) {}
  };
  class A {
   public:
   private:
    friend class vector<A>;
    A() {}
  };
  void f() {
    vector<A> v(1);
  }
}
namespace test6_3 {
  template<class T>
  class vector {
   public:
    vector(int i) {}
    void f(const T& t = T()) {}
  };
  class A {
   public:
   private:
    friend void vector<A>::f(const A&);
    A() {}
  };
  void f() {
    vector<A> v(1);
    v.f();
  }
}

namespace test7 {
  extern "C" {
    class X {
      friend int f() { return 42; }
    };
  }
}

// PR15485
namespace test8 {
  namespace ns1 {
    namespace ns2 {
      template<class T> void f(T t); // expected-note {{target of using declaration}}
    }
    using ns2::f; // expected-note {{using declaration}}
  }
  struct A { void f(); }; // expected-note {{target of using declaration}}
  struct B : public A { using A::f; }; // expected-note {{using declaration}}
  struct X {
    template<class T> friend void ns1::f(T t); // expected-error {{cannot befriend target of using declaration}}
    friend void B::f(); // expected-error {{cannot befriend target of using declaration}}
  };
}

// PR16423
namespace test9 {
  class C {
  };
  struct A {
    friend void C::f(int, int, int) {}  // expected-error {{no function named 'f' with type 'void (int, int, int)' was found in the specified scope}}
  };
}
