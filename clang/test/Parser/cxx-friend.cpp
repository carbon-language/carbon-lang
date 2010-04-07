// RUN: %clang_cc1 -fsyntax-only -verify %s

class C {
  friend class D;
};

class A {
public:
  void f();
};

friend int x; // expected-error {{'friend' used outside of class}}

friend class D {}; // expected-error {{'friend' used outside of class}}

union U {
  int u1;
};

class B {
  // 'A' here should refer to the declaration above.  
  friend class A;

  friend C; // expected-warning {{must specify 'class' to befriend}}
  friend U; // expected-warning {{must specify 'union' to befriend}}
  friend int; // expected-warning {{non-class type 'int' cannot be a friend}}

  friend void myfunc();

  void f(A *a) { a->f(); }
};





template <typename t1, typename t2> class some_template;
friend   // expected-error {{'friend' used outside of class}}
some_template<foo, bar>&  // expected-error {{use of undeclared identifier 'foo'}}
  ;  // expected-error {{expected unqualified-id}}
