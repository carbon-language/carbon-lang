// RUN: clang -fsyntax-only -verify %s
void f() {
  int *ptr = malloc(sizeof(int) * 10); // expected-warning{{implicitly declaring C library function 'malloc' with type}} \
  // expected-note{{please include the header <stdlib.h> or explicitly provide a declaration for 'malloc'}} \
  // expected-note{{'malloc' is a builtin with type 'void *}}
}

void *alloca(__SIZE_TYPE__); // redeclaration okay

int *calloc(__SIZE_TYPE__, __SIZE_TYPE__); // expected-warning{{incompatible redeclaration of library function 'calloc' will be ignored}} \
                    // expected-note{{'calloc' is a builtin with type 'void *}}


void g(int malloc) { // okay: these aren't functions
  int calloc = 1;
}

void h() {
  int malloc(int); // expected-warning{{incompatible redeclaration of library function 'malloc' will be ignored}}
  int strcpy(int); // expected-warning{{incompatible redeclaration of library function 'strcpy' will be ignored}} \
  // expected-note{{'strcpy' is a builtin with type 'char *(char *, char const *)'}}
}

void f2() {
  fprintf(0, "foo"); // expected-error{{implicit declaration of 'fprintf' requires inclusion of the header <stdio.h>}}
}

// PR2892
void __builtin_object_size(); // expected-error{{conflicting types}} \
// expected-note{{'__builtin_object_size' is a builtin with type}}

int a[10];

int f0() {
  return __builtin_object_size(&a); // expected-error {{too few arguments to function}}
}

void * realloc(void *p, int size) { // expected-warning{{incompatible redeclaration of library function 'realloc' will be ignored}} \
// expected-note{{'realloc' is a builtin with type 'void *(void *,}}
  return p;
}
