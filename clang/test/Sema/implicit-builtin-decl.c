// RUN: clang -fsyntax-only -verify %s
void f() {
  int *ptr = malloc(sizeof(int) * 10); // expected-warning{{implicitly declaring C library function 'malloc' with type}} \
  // expected-note{{please include the header <stdlib.h> or explicitly provide a declaration for 'malloc'}} \
  // expected-note{{'malloc' was implicitly declared here with type 'void *}}
}

void *alloca(__SIZE_TYPE__); // redeclaration okay

int *calloc(__SIZE_TYPE__, __SIZE_TYPE__); // expected-error{{conflicting types for 'calloc'}} \
                    // expected-note{{'calloc' was implicitly declared here with type 'void *}}


void g(int malloc) { // okay: these aren't functions
  int calloc = 1;
}

void h() {
  int malloc(int); // expected-error{{conflicting types for 'malloc'}}
  int strcpy(int); // expected-error{{conflicting types for 'strcpy'}} \
  // expected-note{{'strcpy' was implicitly declared here with type 'char *(char *, char const *)'}}
}

void f2() {
  fprintf(0, "foo"); // expected-error{{implicit declaration of 'fprintf' requires inclusion of the header <stdio.h>}}
}
