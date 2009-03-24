// RUN: clang-cc -fsyntax-only -verify %s

void f5 (int z) { 
  if (z) 
    default:  // expected-error {{not in switch statement}}
      ; // expected-warning {{if statement has empty body}}
} 

