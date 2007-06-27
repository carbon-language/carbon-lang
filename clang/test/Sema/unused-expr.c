// RUN: clang -parse-ast-check %s

int foo(int X, int Y);

void bar(volatile int *VP, int *P, int A,
         _Complex double C, volatile _Complex double VC) {
  
  VP == P;             // expected-warning {{expression result unused}}
  (void)A;             // expected-warning {{expression result unused}}
  (void)foo(1,2);      // no warning.
  
  A == foo(1, 2);      // expected-warning {{expression result unused}}

  foo(1,2)+foo(4,3);   // expected-warning {{expression result unused}}


  *P;                  // expected-warning {{expression result unused}}
  *VP;                 // no warning.
  P[4];                // expected-warning {{expression result unused}}
  VP[4];               // no warning.

  // FIXME: SEMA explodes on these.
  //__real__ C;
  //__real__ VC;
}

