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
  class A{
  private:
    ~A(); // expected-note 3 {{declared private here}}
    static A foo;
  };

  A a; // expected-error {{'~A' is a private member}}
  A A::foo;

  void foo(A param) { // expected-error {{'~A' is a private member}}
    A local; // expected-error {{'~A' is a private member}}
  }
}
