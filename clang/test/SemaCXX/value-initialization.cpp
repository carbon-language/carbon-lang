// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

struct A { // expected-error {{implicit default constructor for 'A' must explicitly initialize the const member 'i'}} \
  // expected-warning{{struct 'A' does not declare any constructor to initialize its non-modifiable members}}
     const int i;	// expected-note {{declared here}} \
  // expected-note{{const member 'i' will never be initialized}}
     virtual void f() { } 
};

int main () {
      (void)A(); // expected-note {{first required here}}
}
