// RUN: %clang_cc1 -std=c++11 -verify %s
struct A { // expected-note 2{{candidate}}
  A(int); // expected-note {{candidate}}
  int n;
};
int a = A().n; // expected-error {{no matching constructor}}

struct B {
  B() = delete; // expected-note {{here}}
  int n;
};
int b = B().n; // expected-error {{call to deleted}}

struct C { // expected-note {{here}}
  B b;
};
int c = C().b.n; // expected-error {{call to implicitly-deleted default}}

struct D { // expected-note {{defined here}}
  D() = default; // expected-note {{declared here}}
  B b;
};
int d = D().b.n; // expected-error {{call to implicitly-deleted default}}

struct E {
  E() = default;
  int n;
};
int e = E().n; // ok

struct F {
  F();
  int n;
};
int f = F().n; // ok

union G { // expected-note {{here}}
  F f;
};
int g = G().f.n; // expected-error {{call to implicitly-deleted default}}

struct H {
  int n;
private:
  H(); // expected-note {{here}}
};
int h = H().n; // expected-error {{private constructor}}

struct I { // expected-note {{here}}
  H h;
};
int i = I().h.n; // expected-error {{call to implicitly-deleted default}}

struct J {
  J();
  virtual int f();
  int n;
};
int j1 = J().n; // ok
int j2 = J().f(); // ok

union K { // expected-note 2{{here}}
  J j;
  int m;
};
int k1 = K().j.n; // expected-error {{call to implicitly-deleted default}}
int k2 = K().j.f(); // expected-error {{call to implicitly-deleted default}}
