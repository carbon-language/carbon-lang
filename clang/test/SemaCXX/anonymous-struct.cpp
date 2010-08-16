// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S {
  S();  // expected-note {{because type 'S' has a user-declared constructor}}    
};

struct E {
  struct {
    S x;  // expected-error {{anonymous struct member 'x' has a non-trivial constructor}} 
  };
};
