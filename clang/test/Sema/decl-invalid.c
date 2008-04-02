// RUN: clang %s -fsyntax-only -verify

typedef union <anonymous> __mbstate_t;  // expected-error: {{expected identifier or}}


// PR2017
void x(); 
int a() {
  int r[x()];  // expected-error: {{size of array has non-integer type 'void'}}
}

