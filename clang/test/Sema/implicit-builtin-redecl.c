// RUN: clang -fsyntax-only -verify %s

// PR3592
static void* malloc(int);
static void* malloc(int size) {
  return ((void*)0); /*do not use heap in this file*/
}

void *calloc(int, int, int); // expected-warning{{incompatible redeclaration of library function 'calloc' will be ignored}} \
// expected-note{{'calloc' is a builtin with type 'void *(unsigned long, unsigned long)'}}

void f1(void) { 
  return calloc(0, 0, 0);  // expected-error{{too many arguments to function call}}
}
