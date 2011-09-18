// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

int n;
struct S {
  int &a; // expected-note 2{{here}}
  int &b = n;

  union {
    const int k = 42;
  };

  S() {} // expected-error {{constructor for 'S' must explicitly initialize the reference member 'a'}}
  S(int) : a(n) {} // ok
  S(char) : b(n) {} // expected-error {{constructor for 'S' must explicitly initialize the reference member 'a'}}
  S(double) : a(n), b(n) {} // ok
} s(0);

union U {
  int a = 0;
  char b = 'x';

  // FIXME: these should all be rejected
  U() {} // desired-error {{at most one member of a union may be initialized}}
  U(int) : a(1) {} // desired-error {{at most one member of a union may be initialized}}
  U(char) : b('y') {} // desired-error {{at most one member of a union may be initialized}}
  U(double) : a(1), b('y') {} // desired-error {{at most one member of a union may be initialized}}
};

// PR10954: variant members do not acquire an implicit initializer.
namespace VariantMembers {
  struct NoDefaultCtor {
    NoDefaultCtor(int);
  };
  union V {
    NoDefaultCtor ndc;
    int n;

    V() {}
    V(int n) : n(n) {}
    V(int n, bool) : ndc(n) {}
  };
  struct K {
    union {
      NoDefaultCtor ndc;
      int n;
    };
    K() {}
    K(int n) : n(n) {}
    K(int n, bool) : ndc(n) {}
  };
}
