// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S {
  S();  // expected-note {{because type 'S' has a user-provided default constructor}}
};

struct { // expected-error {{anonymous structs and classes must be class members}}
};

struct E {
  struct {
    S x;  // expected-error {{anonymous struct member 'x' has a non-trivial constructor}}
  };
  static struct {
  };
};

template <class T> void foo(T);
typedef struct { // expected-note {{use a tag name here to establish linkage prior to definition}} expected-note {{declared here}}
  void test() {
    foo(this); // expected-warning {{template argument uses unnamed type}}
  }
} A; // expected-error {{unsupported: typedef changes linkage of anonymous type, but linkage was already computed}}
