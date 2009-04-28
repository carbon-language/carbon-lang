// RUN: clang-cc -fsyntax-only -verify -pedantic -fblocks %s

// PR2241
float test2241[2] = { 
  1e,            // expected-error {{exponent}}
  1ee0           // expected-error {{exponent}}
};


// Testcase derived from PR2692
static char *f (char * (*g) (char **, int), char **p, ...) {
    char *s;
    va_list v;                              // expected-error {{identifier}}
    s = g (p, __builtin_va_arg(v, int));    // expected-error {{identifier}}
}


// PR3172
} // expected-error {{expected external declaration}}


// rdar://6094870
int test(int a) {
  struct { int i; } x;
  
  if (x.hello)   // expected-error {{no member named 'hello'}}
    test(0);
  else
    ;
  
  if (x.hello == 0)   // expected-error {{no member named 'hello'}}
    test(0);
  else
    ;
  
  if ((x.hello == 0))   // expected-error {{no member named 'hello'}}
    test(0);
  else
    ;
  
  if (x.i == 0))   // expected-error {{expected expression}}
    test(0);
  else
    ;
}



char ((((                       /* expected-note {{to match this '('}} */
         *X x ] ))));                    /* expected-error {{expected ')'}} */

;   // expected-warning {{ISO C does not allow an extra ';' outside of a function}}




struct S { void *X, *Y; };

struct S A = {
&BADIDENT, 0     /* expected-error {{use of undeclared identifier}} */
};

// rdar://6248081
int test6248081() { 
  [10]  // expected-error {{expected expression}}
}

struct forward; // expected-note{{forward declaration of 'struct forward'}}
void x(struct forward* x) {switch(x->a) {}} // expected-error {{incomplete definition of type}}

// PR3410
void foo() {
  int X;
  X = 4 // expected-error{{expected ';' after expression}}
}
