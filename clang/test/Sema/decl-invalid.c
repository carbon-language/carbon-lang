// RUN: clang %s -fsyntax-only -verify

typedef union <anonymous> __mbstate_t;  // expected-error: {{declaration of anonymous union must be a definition}}


// PR2017
void x(); 
int a() {
  int r[x()];  // expected-error: {{size of array has non-integer type 'void'}}
}

