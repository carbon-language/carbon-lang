// RUN: %clang_cc1 -fsyntax-only -verify %s

void (*e) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (1,1)));

int main() {
  void (*b) (int arg, const char * format, ...) __attribute__ ((__sentinel__));  // expected-note {{function has been explicitly marked sentinel here}}
  void (*z) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (2))); // expected-note {{function has been explicitly marked sentinel here}}


  void (*y) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (5))); // expected-note {{function has been explicitly marked sentinel here}}

  b(1, "%s", (void*)0); // OK
  b(1, "%s", 0);  // expected-warning {{missing sentinel in function call}}
  z(1, "%s",4 ,1,0);  // expected-warning {{missing sentinel in function call}}
  z(1, "%s", (void*)0, 1, 0); // OK

  y(1, "%s", 1,2,3,4,5,6,7);  // expected-warning {{missing sentinel in function call}}

  y(1, "%s", (void*)0,3,4,5,6,7); // OK
}
