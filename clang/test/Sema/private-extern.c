// RUN: clang-cc -verify -fsyntax-only %s

static int g0; // expected-note{{previous definition}}
int g0; // expected-error{{non-static declaration of 'g0' follows static declaration}}

static int g1;
extern int g1;

static int g2; 
__private_extern__ int g2;

int g3; // expected-note{{previous definition}}
static int g3; // expected-error{{static declaration of 'g3' follows non-static declaration}}

extern int g4; // expected-note{{previous definition}}
static int g4; // expected-error{{static declaration of 'g4' follows non-static declaration}}

__private_extern__ int g5; // expected-note{{previous definition}}
static int g5; // expected-error{{static declaration of 'g5' follows non-static declaration}}

void f0() {
  // FIXME: Diagnose this?
  int g6;
  extern int g6;
}

void f1() {
  // FIXME: Diagnose this?
  int g7;
  __private_extern__ int g7;
}

void f2() {
  extern int g8; // expected-note{{previous definition}}
  // FIXME: Improve this diagnostic.
  int g8; // expected-error{{redefinition of 'g8'}}
}

void f3() {
  __private_extern__ int g9; // expected-note{{previous definition}}
  // FIXME: Improve this diagnostic.
  int g9; // expected-error{{redefinition of 'g9'}}
}

void f4() {
  extern int g10;
  extern int g10;
}

void f5() {
  __private_extern__ int g11;
  __private_extern__ int g11;
}

void f6() {
  // FIXME: Diagnose
  extern int g12;
  __private_extern__ int g12;
}

void f7() {
  // FIXME: Diagnose
  __private_extern__ int g13;
  extern int g13;
}

struct s0;
void f8() {
  extern struct s0 g14;
  __private_extern__ struct s0 g14;
}
struct s0 { int x; };

void f9() {
  extern int g15 = 0; // expected-error{{'extern' variable cannot have an initializer}}
  // FIXME: linkage specifier in warning.
  __private_extern__ int g16 = 0; // expected-error{{'extern' variable cannot have an initializer}}
}

extern int g17;
int g17 = 0;

extern int g18 = 0; // expected-warning{{'extern' variable has an initializer}}

__private_extern__ int g19;
int g19 = 0;

__private_extern__ int g20 = 0;
