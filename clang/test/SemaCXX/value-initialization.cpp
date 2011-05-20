// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

struct A { //expected-note {{marked deleted here}} \
     // expected-warning {{does not declare any constructor to initialize}}
     const int i; // expected-note{{const member 'i' will never be initialized}}
     virtual void f() { } 
};

int main () {
      (void)A(); // expected-error {{call to deleted constructor}}
}
