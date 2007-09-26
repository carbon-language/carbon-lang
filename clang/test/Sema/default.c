// RUN: clang -parse-ast -verify %s

void f5 (int z) { 
  if (z) 
    default:  // expected-error {{not in switch statement}}
      ; 
} 

