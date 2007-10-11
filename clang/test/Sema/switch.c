// RUN: clang -fsyntax-only -verify %s

void f (int z) { 
  while (z) { 
    default: z--;            // expected-error {{statement not in switch}}
  } 
}

void foo(int X) {
  switch (X) {
  case 42: ;                 // expected-error {{previous case value}}
  case 5000000000LL:         // expected-warning {{overflow}}
  case 42:                   // expected-error {{duplicate case value}}
   ;

  case 100 ... 99: ;         // expected-warning {{empty case range}}

  case 43: ;                 // expected-error {{previous case value}}
  case 43 ... 45:  ;         // expected-error {{duplicate case value}}

  case 100 ... 20000:;       // expected-error {{previous case value}}
  case 15000 ... 40000000:;  // expected-error {{duplicate case value}}
  }
}

void test3(void) { 
  // empty switch;
  switch (0); 
}

