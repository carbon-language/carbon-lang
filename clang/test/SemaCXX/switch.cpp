// RUN: clang-cc -fsyntax-only -verify %s

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
