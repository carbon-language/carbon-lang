// RUN: clang-cc -fsyntax-only -verify -std=c++0x %s

// PR5518
struct A { 
  explicit operator int(); // expected-note{{conversion to integral type}}
};

void x() { 
  switch(A()) { // expected-error{{explicit conversion to}}
  }
}
