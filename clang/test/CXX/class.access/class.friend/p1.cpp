// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++'0x [class.friend] p1:
//   A friend of a class is a function or class that is given permission to use
//   the private and protected member names from the class. A class specifies
//   its friends, if any, by way of friend declarations. Such declarations give
//   special access rights to the friends, but they do not make the nominated
//   friends members of the befriending class.
//
// FIXME: Add tests for access control when implemented. Currently we only test
// for parsing.

struct S { static void f(); };
S* g() { return 0; }

struct X {
  friend struct S;
  friend S* g();
};

void test1() {
  S s;
  g()->f();
  S::f();
  X::g(); // expected-error{{no member named 'g' in 'X'}}
  X::S x_s; // expected-error{{no member named 'S' in 'X'}}
  X x;
  x.g(); // expected-error{{no member named 'g' in 'X'}}
}

// Test that we recurse through namespaces to find already declared names, but
// new names are declared within the enclosing namespace.
namespace N {
  struct X {
    friend struct S;
    friend S* g();

    friend struct S2;
    friend struct S2* g2();
  };

  struct S2 { static void f2(); };
  S2* g2() { return 0; }

  void test() {
    g()->f();
    S s;
    S::f();
    X::g(); // expected-error{{no member named 'g' in 'N::X'}}
    X::S x_s; // expected-error{{no member named 'S' in 'N::X'}}
    X x;
    x.g(); // expected-error{{no member named 'g' in 'N::X'}}

    g2();
    S2 s2;
    ::g2(); // expected-error{{no member named 'g2' in the global namespace}}
    ::S2 g_s2; // expected-error{{no member named 'S2' in the global namespace}}
    X::g2(); // expected-error{{no member named 'g2' in 'N::X'}}
    X::S2 x_s2; // expected-error{{no member named 'S2' in 'N::X'}}
    x.g2(); // expected-error{{no member named 'g2' in 'N::X'}}
  }
}
