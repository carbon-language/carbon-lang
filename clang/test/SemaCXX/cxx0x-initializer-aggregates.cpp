// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct one { char c[1]; };
struct two { char c[2]; };

namespace aggregate {
  // Direct list initialization does NOT allow braces to be elided!
  struct S {
    int ar[2];
    struct T {
      int i1;
      int i2;
    } t;
    struct U {
      int i1;
    } u[2];
    struct V {
      int var[2];
    } v;
  };

  void bracing() {
    S s1 = { 1, 2, 3 ,4, 5, 6, 7, 8 }; // no-error
    S s2{ {1, 2}, {3, 4}, { {5}, {6} }, { {7, 8} } }; // completely braced
    S s3{ 1, 2, 3, 4, 5, 6 }; // expected-error 5 {{cannot omit braces}}
    S s4{ {1, 2}, {3, 4}, {5, 6}, { {7, 8} } }; // expected-error 2 {{cannot omit braces}}
    S s5{ {1, 2}, {3, 4}, { {5}, {6} }, {7, 8} }; // expected-error {{cannot omit braces}}
  }

  struct String {
    String(const char*);
  };

  struct A {
    int m1;
    int m2;
  };

  void function_call() {
    void takes_A(A);
    takes_A({1, 2});
  }

  struct B {
    int m1;
    String m2;
  };

  void overloaded_call() {
    one overloaded(A);
    two overloaded(B);

    static_assert(sizeof(overloaded({1, 2})) == sizeof(one), "bad overload");
    static_assert(sizeof(overloaded({1, "two"})) == sizeof(two),
      "bad overload");
    // String is not default-constructible
    static_assert(sizeof(overloaded({1})) == sizeof(one), "bad overload");
  }
}
