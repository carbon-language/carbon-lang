// RUN: clang -fsyntax-only -verify -pedantic %s
void foo() {
  *(0 ? (double *)0 : (void *)0) = 0;
  *(0 ? (double *)0 : (void *)(int *)0) = 0;
  *(0 ? (double *)0 : (void *)(double *)0) = 0;
  *(0 ? (double *)0 : (int *)(void *)0) = 0; // expected-warning {{pointer type mismatch ('double *' and 'int *')}}
  *(0 ? (double *)0 : (double *)(void *)0) = 0;
  *((void *) 0) = 0; // expected-error {{incomplete type 'void' is not assignable}}
  double *dp;
  int *ip;
  void *vp;

  dp = vp;
  vp = dp;
  ip = dp; // expected-warning {{incompatible pointer types assigning 'double *', expected 'int *'}}
  dp = ip; // expected-warning {{incompatible pointer types assigning 'int *', expected 'double *'}}
  dp = 0 ? (double *)0 : (void *)0;
  vp = 0 ? (double *)0 : (void *)0;
  ip = 0 ? (double *)0 : (void *)0; // expected-warning {{incompatible pointer types assigning 'double *', expected 'int *'}}
}

