// RUN: clang -fsyntax-only -verify -pedantic %s


// PR2241
float f[] = { 
  1e,            // expected-error {{exponent}}
  1ee0           // expected-error {{exponent}}
};
