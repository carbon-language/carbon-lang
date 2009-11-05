// RUN: clang-cc -fsyntax-only -pedantic -verify %s

int test(char *C) { // nothing here should warn.
  return C != ((void*)0);
  return C != (void*)0;
  return C != 0;  
  return C != 1;  // expected-warning {{comparison between pointer and integer ('char *' and 'int')}}
}

int ints(long a, unsigned long b) {
  return (a == b) +        // expected-warning {{comparison of integers of different signs}}
         ((int)a == b) +   // expected-warning {{comparison of integers of different signs}}
         ((short)a == b) + // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned short) b); // expected-warning {{comparison of integers of different signs}}
}

int equal(char *a, const char *b) {
    return a == b;
}

int arrays(char (*a)[5], char(*b)[10], char(*c)[5]) {
  int d = (a == c);
  return a == b; // expected-warning {{comparison of distinct pointer types}}
}

int pointers(int *a) {
  return a > 0; // expected-warning {{ordered comparison between pointer and zero ('int *' and 'int') is an extension}}
  return a > 42; // expected-warning {{ordered comparison between pointer and integer ('int *' and 'int')}}
  return a > (void *)0; // expected-warning {{comparison of distinct pointer types}}
}

int function_pointers(int (*a)(int), int (*b)(int), void (*c)(int)) {
  return a > b; // expected-warning {{ordered comparison of function pointers}}
  return function_pointers > function_pointers; // expected-warning {{ordered comparison of function pointers}}
  return a > c; // expected-warning {{comparison of distinct pointer types}}
  return a == (void *) 0;
  return a == (void *) 1; // expected-warning {{equality comparison between function pointer and void pointer}}
}

int void_pointers(void* foo) {
  return foo == (void*) 0;
  return foo == (void*) 1;
}
