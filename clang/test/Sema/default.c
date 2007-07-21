// RUN: clang -parse-ast-check %s

void f5 (int z) { 
  if (z) 
    default:  // expected-error {{not in switch statement}}
      ; 
} 

