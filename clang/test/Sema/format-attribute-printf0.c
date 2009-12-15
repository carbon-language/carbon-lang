//RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdarg.h>

// same as format(printf(...))...
void a2(const char *a, ...)    __attribute__((format(printf0, 1,2))); // no-error
void b2(const char *a, ...)    __attribute__((format(printf0, 1,1))); // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c2(const char *a, ...)    __attribute__((format(printf0, 0,2))); // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d2(const char *a, int c)  __attribute__((format(printf0, 1,2))); // expected-error {{format attribute requires variadic function}}
void e2(char *str, int c, ...) __attribute__((format(printf0, 2,3))); // expected-error {{format argument not a string type}}

// FreeBSD usage
#define __printf0like(fmt,va) __attribute__((__format__(__printf0__,fmt,va)))
void null(int i, const char *a, ...) __printf0like(2,0); // no-error
void null(int i, const char *a, ...) {
  if (a)
    (void)0/* vprintf(...) would go here */;
}

void callnull(void){
  null(0,        0); // no error
  null(0, (char*)0); // no error
  null(0, (void*)0); // no error
  null(0,  (int*)0); // expected-warning {{incompatible pointer types}}
}

