// RUN: clang -parse-ast-check %s

void f (int z) { 
  while (z) { 
    default: z--;   // expected-error {{statement not in switch}}
  } 
}

void foo(int X) {
  switch (X) {
  case 42: ;          // expected-error {{previous case value}}
  case 5000000000LL:  // expected-warning {{overflow}}
  case 42:            // expected-error {{duplicate case value}}
   ;
  }
}

