// RUN: clang -parse-ast-check %s


void f (int z) { 
  while (z) { 
    default: z--;   // expected-error {{statement not in switch}}
  } 
}

