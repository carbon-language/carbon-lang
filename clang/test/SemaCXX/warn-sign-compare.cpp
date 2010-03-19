// RUN: %clang_cc1 -verify -fsyntax-only -Wsign-compare %s

// NOTE: When a 'enumeral mismatch' warning is implemented then expect several
// of the following cases to be impacted.

// namespace for anonymous enums tests
namespace test1 {
  enum { A };
  enum { B = -1 };

  template <typename T> struct Foo {
    enum { C };
    enum { D = ~0U };
  };

  enum { E = ~0U };

  void doit_anonymous( int i ) {
    int a1 = 1 ? i : A;
    int a2 = 1 ? A : i;

    int b1 = 1 ? i : B;
    int b2 = 1 ? B : i;

    int c1 = 1 ? i : Foo<bool>::C;
    int c2 = 1 ? Foo<bool>::C : i;

    int d1 = 1 ? i : Foo<bool>::D; // expected-warning {{operands of ? are integers of different signs}}
    int d2 = 1 ? Foo<bool>::D : i; // expected-warning {{operands of ? are integers of different signs}}
    int d3 = 1 ? B : Foo<bool>::D; // expected-warning {{operands of ? are integers of different signs}}
    int d4 = 1 ? Foo<bool>::D : B; // expected-warning {{operands of ? are integers of different signs}}

    int e1 = 1 ? i : E; // expected-warning {{operands of ? are integers of different signs}}
    int e2 = 1 ? E : i; // expected-warning {{operands of ? are integers of different signs}}
    int e3 = 1 ? E : B; // expected-warning {{operands of ? are integers of different signs}}
    int e4 = 1 ? B : E; // expected-warning {{operands of ? are integers of different signs}}
  }
}

// namespace for named enums tests
namespace test2 {
  enum Named1 { A };
  enum Named2 { B = -1 };

  template <typename T> struct Foo {
    enum Named3 { C };
    enum Named4 { D = ~0U };
  };

  enum Named5 { E = ~0U };

  void doit_anonymous( int i ) {
    int a1 = 1 ? i : A;
    int a2 = 1 ? A : i;

    int b1 = 1 ? i : B;
    int b2 = 1 ? B : i;

    int c1 = 1 ? i : Foo<bool>::C;
    int c2 = 1 ? Foo<bool>::C : i;

    int d1 = 1 ? i : Foo<bool>::D; // expected-warning {{operands of ? are integers of different signs}}
    int d2 = 1 ? Foo<bool>::D : i; // expected-warning {{operands of ? are integers of different signs}}
    int d3 = 1 ? B : Foo<bool>::D; // expected-warning {{operands of ? are integers of different signs}}
    int d4 = 1 ? Foo<bool>::D : B; // expected-warning {{operands of ? are integers of different signs}}

    int e1 = 1 ? i : E; // expected-warning {{operands of ? are integers of different signs}}
    int e2 = 1 ? E : i; // expected-warning {{operands of ? are integers of different signs}}
    int e3 = 1 ? E : B; // expected-warning {{operands of ? are integers of different signs}}
    int e4 = 1 ? B : E; // expected-warning {{operands of ? are integers of different signs}}
  }
}
