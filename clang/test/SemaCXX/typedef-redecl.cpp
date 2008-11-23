// RUN: clang -fsyntax-only -verify %s 

typedef int INT;
typedef INT REALLY_INT; // expected-note {{previous definition is here}}
typedef REALLY_INT REALLY_REALLY_INT;
typedef REALLY_INT BOB;
typedef float REALLY_INT; // expected-error{{typedef redefinition with different types ('float' vs 'INT')}}

class X {
  typedef int result_type; // expected-note {{previous definition is here}}
  typedef INT result_type; // expected-error {{redefinition of 'result_type'}}
};
