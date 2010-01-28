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
    op->foo(ProtectedInst); // expected-error {{access to protected member outside any class}}
    op->foo(PrivateInst); // expected-error {{access to private member outside any class}}

    void (A::*a)(Public&) = &A::foo;
    void (A::*b)(Protected&) = &A::foo; // expected-error {{access to protected member outside any class}}
    void (A::*c)(Private&) = &A::foo; // expected-error {{access to private member outside any class}}
  }
}

// Member operators.
namespace test1 {
  class A {
  public:
    void operator+(Public&);
    void operator[](Public&);
    void operator()(Public&);
  protected:
    void operator+(Protected&); // expected-note {{declared protected here}}
    void operator[](Protected&); // expected-note {{declared protected here}}
    void operator()(Protected&); // expected-note {{declared protected here}}
  private:
    void operator+(Private&); // expected-note {{declared private here}}
    void operator[](Private&); // expected-note {{declared private here}}
    void operator()(Private&); // expected-note {{declared private here}}
    void operator-(); // expected-note {{declared private here}}
  };
  void operator+(const A &, Public&);
  void operator+(const A &, Protected&);
  void operator+(const A &, Private&);
  void operator-(const A &);

  void test(A &a, Public &pub, Protected &prot, Private &priv) {
    a + pub;
    a + prot; // expected-error {{access to protected member}}
    a + priv; // expected-error {{access to private member}}
    a[pub];
    a[prot]; // expected-error {{access to protected member}}
    a[priv]; // expected-error {{access to private member}}
    a(pub);
    a(prot); // expected-error {{access to protected member}}
    a(priv); // expected-error {{access to private member}}
    -a;       // expected-error {{access to private member}}

    const A &ca = a;
    ca + pub;
    ca + prot;
    ca + priv;
    -ca;
  }
}
