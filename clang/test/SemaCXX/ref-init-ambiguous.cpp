// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

enum E2 { };

struct A { 
  operator E2&(); // expected-note 3 {{candidate function}}
};

struct B { 
  operator E2&(); // expected-note 3 {{candidate function}}
};

struct C : B, A { 
};

void test(C c) {
  const E2 &e2 = c; // expected-error {{reference initialization of type 'enum E2 const &' with initializer of type 'struct C' is ambiguous}}
}

void foo(const E2 &);

const E2 & re(C c) {
    foo(c); // expected-error {{reference initialization of type 'enum E2 const &' with initializer of type 'struct C' is ambiguous}}

    return c; // expected-error {{reference initialization of type 'enum E2 const &' with initializer of type 'struct C' is ambiguous}}
}


