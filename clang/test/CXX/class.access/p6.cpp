// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++0x [class.access]p6:
//   All access controls in [class.access] affect the ability to
//   access a class member name from a particular scope. For purposes
//   of access control, the base-specifiers of a class and the
//   definitions of class members that appear outside of the class
//   definition are considered to be within the scope of that
//   class. In particular, access controls apply as usual to member
//   names accessed as part of a function return type, even though it
//   is not possible to determine the access privileges of that use
//   without first parsing the rest of the function
//   declarator. Similarly, access control for implicit calls to the
//   constructors, the conversion functions, or the destructor called
//   to create and destroy a static data member is performed as if
//   these calls appeared in the scope of the member's class.

struct Public {}; struct Protected {}; struct Private {};

namespace test0 {
  class A {
    typedef int type; // expected-note {{declared private here}}
    type foo();
  };

  A::type foo() { } // expected-error {{'type' is a private member}}
  A::type A::foo() { }
}

// conversion decls
namespace test1 {
  class A {
  public:
    A();
    operator Public ();
    A(Public);
  protected:
    operator Protected (); // expected-note {{declared protected here}}
    A(Protected); // expected-note {{declared protected here}}
  private:
    operator Private (); // expected-note {{declared private here}}
    A(Private); // expected-note {{declared private here}}
  };

  void test() {
    A a;
    Public pub = a;
    Protected prot = a; // expected-error {{'operator Protected' is a protected member}}
    Private priv = a; // expected-error {{'operator Private' is a private member}}
    A apub = pub;
    A aprot = prot; // expected-error {{protected constructor}}
    A apriv = priv; // expected-error {{private constructor}}
  }
}
