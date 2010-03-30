// RUN: %clang_cc1 -fsyntax-only -faccess-control -verify %s

// C++0x [class.access]p4:

//   Access control is applied uniformly to all names, whether the
//   names are referred to from declarations or expressions.  In the
//   case of overloaded function names, access control is applied to
//   the function selected by overload resolution.

class Public {} PublicInst;
class Protected {} ProtectedInst;
class Private {} PrivateInst;

namespace test0 {
  class A {
  public:
    void foo(Public&);
  protected:
    void foo(Protected&); // expected-note 2 {{declared protected here}}
  private:
    void foo(Private&); // expected-note 2 {{declared private here}}
  };

  void test(A *op) {
    op->foo(PublicInst);
    op->foo(ProtectedInst); // expected-error {{'foo' is a protected member}}
    op->foo(PrivateInst); // expected-error {{'foo' is a private member}}

    void (A::*a)(Public&) = &A::foo;
    void (A::*b)(Protected&) = &A::foo; // expected-error {{'foo' is a protected member}}
    void (A::*c)(Private&) = &A::foo; // expected-error {{'foo' is a private member}}
  }
}

// Member operators.
namespace test1 {
  class A {
  public:
    void operator+(Public&);
    void operator[](Public&);
    void operator()(Public&);
    typedef void (*PublicSurrogate)(Public&);
    operator PublicSurrogate() const;
  protected:
    void operator+(Protected&); // expected-note {{declared protected here}}
    void operator[](Protected&); // expected-note {{declared protected here}}
    void operator()(Protected&); // expected-note {{declared protected here}}
    typedef void (*ProtectedSurrogate)(Protected&);
    operator ProtectedSurrogate() const; // expected-note {{declared protected here}}
  private:
    void operator+(Private&); // expected-note {{declared private here}}
    void operator[](Private&); // expected-note {{declared private here}}
    void operator()(Private&); // expected-note {{declared private here}}
    void operator-(); // expected-note {{declared private here}}
    typedef void (*PrivateSurrogate)(Private&);
    operator PrivateSurrogate() const; // expected-note {{declared private here}}
  };
  void operator+(const A &, Public&);
  void operator+(const A &, Protected&);
  void operator+(const A &, Private&);
  void operator-(const A &);

  void test(A &a, Public &pub, Protected &prot, Private &priv) {
    a + pub;
    a + prot; // expected-error {{'operator+' is a protected member}}
    a + priv; // expected-error {{'operator+' is a private member}}
    a[pub];
    a[prot]; // expected-error {{'operator[]' is a protected member}}
    a[priv]; // expected-error {{'operator[]' is a private member}}
    a(pub);
    a(prot); // expected-error {{'operator()' is a protected member}}
    a(priv); // expected-error {{'operator()' is a private member}}
    -a;       // expected-error {{'operator-' is a private member}}

    const A &ca = a;
    ca + pub;
    ca + prot;
    ca + priv;
    -ca;
    // These are all surrogate calls
    ca(pub);
    ca(prot); // expected-error {{'operator void (*)(class Protected &)' is a protected member}}
    ca(priv); // expected-error {{'operator void (*)(class Private &)' is a private member}}
  }
}

// Implicit constructor calls.
namespace test2 {
  class A {
  private:
    A(); // expected-note {{declared private here}}

    static A foo;
  };

  A a; // expected-error {{calling a private constructor}}
  A A::foo; // okay
}

// Implicit destructor calls.
namespace test3 {
  class A {
  private:
    ~A(); // expected-note 2 {{declared private here}}
    static A foo;
  };

  A a; // expected-error {{variable of type 'test3::A' has private destructor}}
  A A::foo;

  void foo(A param) { // okay
    A local; // expected-error {{variable of type 'test3::A' has private destructor}}
  }

  template <unsigned N> class Base { ~Base(); }; // expected-note 14 {{declared private here}}
  class Base2 : virtual Base<2> { ~Base2(); }; // expected-note 3 {{declared private here}} \
                                               // expected-error {{base class 'Base<2>' has private destructor}}
  class Base3 : virtual Base<3> { public: ~Base3(); }; // expected-error {{base class 'Base<3>' has private destructor}}

  // These don't cause diagnostics because we don't need the destructor.
  class Derived0 : Base<0> { ~Derived0(); };
  class Derived1 : Base<1> { };

  class Derived2 : // expected-error {{inherited virtual base class 'Base<2>' has private destructor}} \
                   // expected-error {{inherited virtual base class 'Base<3>' has private destructor}}
    Base<0>,  // expected-error {{base class 'Base<0>' has private destructor}}
    virtual Base<1>, // expected-error {{base class 'Base<1>' has private destructor}}
    Base2, // expected-error {{base class 'test3::Base2' has private destructor}}
    virtual Base3
  {
    ~Derived2() {}
  };

  class Derived3 : // expected-error 2 {{inherited virtual base class 'Base<2>' has private destructor}} \
                   // expected-error 2 {{inherited virtual base class 'Base<3>' has private destructor}}
    Base<0>,  // expected-error 2 {{base class 'Base<0>' has private destructor}}
    virtual Base<1>, // expected-error 2 {{base class 'Base<1>' has private destructor}}
    Base2, // expected-error 2 {{base class 'test3::Base2' has private destructor}}
    virtual Base3
  {};
  Derived3 d3;
}

// Conversion functions.
namespace test4 {
  class Base {
  private:
    operator Private(); // expected-note 4 {{declared private here}}
  public:
    operator Public();
  };

  class Derived1 : private Base { // expected-note 2 {{declared private here}} \
                                  // expected-note {{constrained by private inheritance}}
    Private test1() { return *this; } // expected-error {{'operator Private' is a private member}}
    Public test2() { return *this; }
  };
  Private test1(Derived1 &d) { return d; } // expected-error {{'operator Private' is a private member}} \
                                           // expected-error {{cannot cast 'test4::Derived1' to its private base class}}
  Public test2(Derived1 &d) { return d; } // expected-error {{cannot cast 'test4::Derived1' to its private base class}} \
                                          // expected-error {{'operator Public' is a private member}}


  class Derived2 : public Base {
    Private test1() { return *this; } // expected-error {{'operator Private' is a private member}}
    Public test2() { return *this; }
  };
  Private test1(Derived2 &d) { return d; } // expected-error {{'operator Private' is a private member}}
  Public test2(Derived2 &d) { return d; }

  class Derived3 : private Base { // expected-note {{constrained by private inheritance here}} \
                                  // expected-note {{declared private here}}
  public:
    operator Private();
  };
  Private test1(Derived3 &d) { return d; }
  Public test2(Derived3 &d) { return d; } // expected-error {{'operator Public' is a private member of 'test4::Base'}} \
                                          // expected-error {{cannot cast 'test4::Derived3' to its private base class}}

  class Derived4 : public Base {
  public:
    operator Private();
  };
  Private test1(Derived4 &d) { return d; }
  Public test2(Derived4 &d) { return d; }
}

// Implicit copy assignment operator uses.
namespace test5 {
  class A {
    void operator=(const A &); // expected-note 2 {{declared private here}}
  };

  class Test1 { A a; }; // expected-error {{field of type 'test5::A' has private copy assignment operator}}
  void test1() {
    Test1 a;
    a = Test1();
  }

  class Test2 : A {}; // expected-error {{base class 'test5::A' has private copy assignment operator}}
  void test2() {
    Test2 a;
    a = Test2();
  }
}

// Implicit copy constructor uses.
namespace test6 {
  class A {
    public: A();
    private: A(const A &); // expected-note 2 {{declared private here}}
  };

  class Test1 { A a; }; // expected-error {{field of type 'test6::A' has private copy constructor}}
  void test1(const Test1 &t) {
    Test1 a = t;
  }

  class Test2 : A {}; // expected-error {{base class 'test6::A' has private copy constructor}}
  void test2(const Test2 &t) {
    Test2 a = t;
  }
}

// Redeclaration lookups are not accesses.
namespace test7 {
  class A {
    int private_member;
  };
  class B : A {
    int foo(int private_member) {
      return 0;
    }
  };
}

// Ignored operator new and delete overloads are not 
namespace test8 {
  typedef __typeof__(sizeof(int)) size_t;

  class A {
    void *operator new(size_t s);
    void operator delete(void *p);
  public:
    void *operator new(size_t s, int n);
    void operator delete(void *p, int n);
  };

  void test() {
    new (2) A();
  }
}

// Don't silently upgrade forbidden-access paths to private.
namespace test9 {
  class A {
    public: static int x;
  };
  class B : private A { // expected-note {{constrained by private inheritance here}}
  };
  class C : public B {
    static int getX() { return x; } // expected-error {{'x' is a private member of 'test9::A'}}
  };
}

namespace test10 {
  class A {
    enum {
      value = 10 // expected-note {{declared private here}}
    };
    friend class C;
  };

  class B {
    enum {
      value = A::value // expected-error {{'value' is a private member of 'test10::A'}}
    };
  };

  class C {
    enum {
      value = A::value
    };
  };
}

namespace test11 {
  class A {
    protected: virtual ~A();
  };

  class B : public A {
    ~B();
  };

  B::~B() {};
}

namespace test12 {
  class A {
    int x;

    void foo() {
      class Local {
        int foo(A *a) {
          return a->x;
        }
      };
    }
  };
}

namespace test13 {
  struct A {
    int x;
    unsigned foo() const;
  };

  struct B : protected A {
    using A::foo;
    using A::x;
  };

  void test() {
    A *d;
    d->foo();
    (void) d->x;
  }
}
