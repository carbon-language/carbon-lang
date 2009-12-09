// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

enum E2 { };

struct A { 
  operator E2&(); // expected-note 2 {{candidate function}}
};

struct B { 
  operator E2&(); // expected-note 2 {{candidate function}}
};

struct C : B, A { 
};

void test(C c) {
  // FIXME: state that there was an ambiguity in the conversion!
  const E2 &e2 = c; // expected-error {{reference to type 'enum E2 const' could not bind to an lvalue of type 'struct C'}}
}

void foo(const E2 &);

const E2 & re(C c) {
    foo(c); // expected-error {{reference initialization of type 'enum E2 const &' with initializer of type 'struct C' is ambiguous}}

    return c; // expected-error {{reference initialization of type 'enum E2 const &' with initializer of type 'struct C' is ambiguous}}
}


