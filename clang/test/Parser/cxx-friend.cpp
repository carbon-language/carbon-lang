// RUN: clang-cc -fsyntax-only -verify %s

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

  friend C; // expected-error {{must specify 'class' in a friend class declaration}}
  friend U; // expected-error {{must specify 'union' in a friend union declaration}}
  friend int; // expected-error {{friends can only be classes or functions}}

  friend void myfunc();

  void f(A *a) { a->f(); }
};

