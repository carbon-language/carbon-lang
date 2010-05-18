// RUN: %clang_cc1 -fsyntax-only -verify -Wswitch-enum %s

void test() {
  bool x = true;
  switch (x) { // expected-warning {{bool}}
    case 0:
      break;
  }

  int n = 3;
  switch (n && 1) { // expected-warning {{bool}}
    case 1:
      break;
  }
}

// PR5518
struct A { 
  operator int(); // expected-note{{conversion to integral type}}
};

void x() { 
  switch(A()) {
  }
}

enum E { e1, e2 };
struct B : A {
  operator E() const; // expected-note{{conversion to enumeration type}}
};

void x2() {
  switch (B()) { // expected-error{{multiple conversions}}
  }
}

struct C; // expected-note{{forward declaration}}

void x3(C &c) {
  switch (c) { // expected-error{{incomplete class type}}
  }
}

namespace test3 {
  enum En { A, B, C };
  template <En how> void foo() {
    int x = 0, y = 5;

    switch (how) { //expected-warning {{no case matching constant switch condition '2'}}
    case A: x *= y; break;
    case B: x += y; break;
    // No case for C, but it's okay because we have a constant condition.
    }
  }

  template void foo<A>();
  template void foo<B>();
  template void foo<C>(); //expected-note {{in instantiation}}
}
