// RUN: clang -fsyntax-only -verify %s

int outer1; // expected-note{{previous definition is here}}
extern int outer2; // expected-note{{previous definition is here}}
int outer4;
int outer4; // expected-note{{previous definition is here}}
int outer5;
int outer6(float); // expected-note{{previous definition is here}}
int outer7(float);

void outer_test() {
  extern float outer1; // expected-error{{redefinition of 'outer1' with a different type}}
  extern float outer2; // expected-error{{redefinition of 'outer2' with a different type}}
  extern float outer3; // expected-note{{previous definition is here}}
  double outer4;
  extern int outer5; // expected-note{{previous definition is here}}
  extern int outer6; // expected-error{{redefinition of 'outer6' as different kind of symbol}}
  int outer7;
  extern int outer8; // expected-note{{previous definition is here}}
  extern int outer9;
  {
    extern int outer9; // expected-note{{previous definition is here}}
  }
}

int outer3; // expected-error{{redefinition of 'outer3' with a different type}}
float outer4; // expected-error{{redefinition of 'outer4' with a different type}}
float outer5;  // expected-error{{redefinition of 'outer5' with a different type}}
int outer8(int); // expected-error{{redefinition of 'outer8' as different kind of symbol}}
float outer9; // expected-error{{redefinition of 'outer9' with a different type}}
