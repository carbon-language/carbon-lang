// RUN: %clang_cc1 -std=c++11 -verify %s
// Per a core issue (no number yet), an ellipsis is always dropped.
struct A {
  A(...); // expected-note {{here}}
  A(int = 0, int = 0, int = 0, int = 0, ...); // expected-note 5{{here}}
  A(int = 0, int = 0, ...); // expected-note {{here}}
};

struct B : A { // expected-note 3{{candidate}}
  using A::A; // expected-warning 3{{inheriting constructor does not inherit ellipsis}} expected-note 4{{candidate}} expected-note 2{{deleted}}
};

B b0{};
// expected-error@-1 {{call to implicitly-deleted default constructor}}
// expected-note@9 {{default constructor of 'B' is implicitly deleted because base class 'A' has multiple default constructors}}

B b1{1};
// FIXME: explain why the inheriting constructor was deleted
// expected-error@-2 {{call to implicitly-deleted function of 'B'}}

B b2{1,2};
// expected-error@-1 {{call to implicitly-deleted function of 'B'}}

B b3{1,2,3};
// ok

B b4{1,2,3,4};
// ok

B b5{1,2,3,4,5};
// expected-error@-1 {{no matching constructor for initialization of 'B'}}
