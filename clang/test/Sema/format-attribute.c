//RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdarg.h>

void a(const char *a, ...) __attribute__((format(printf, 1,2))); // no-error
void b(const char *a, ...) __attribute__((format(printf, 1,1))); // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c(const char *a, ...) __attribute__((format(printf, 0,2))); // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d(const char *a, int c) __attribute__((format(printf, 1,2))); // expected-error {{format attribute requires variadic function}}
void e(char *str, int c, ...) __attribute__((format(printf, 2,3))); // expected-error {{format argument not a string type}}

typedef const char* xpto;
void f(xpto c, va_list list) __attribute__((format(printf, 1, 0))); // no-error
void g(xpto c) __attribute__((format(printf, 1, 0))); // no-error

void y(char *str) __attribute__((format(strftime, 1,0))); // no-error
void z(char *str, int c, ...) __attribute__((format(strftime, 1,2))); // expected-error {{strftime format attribute requires 3rd parameter to be 0}}

int (*f_ptr)(char*,...) __attribute__((format(printf, 1,2))); // no-error
int (*f2_ptr)(double,...) __attribute__((format(printf, 1, 2))); // expected-error {{format argument not a string type}}

struct _mystruct {
  int (*printf)(const char *format, ...) __attribute__((__format__(printf, 1, 2))); // no-error
  int (*printf2)(double format, ...) __attribute__((__format__(printf, 1, 2))); // expected-error {{format argument not a string type}}
};

typedef int (*f3_ptr)(char*,...) __attribute__((format(printf,1,0))); // no-error

// <rdar://problem/6623513>
int rdar6623513(void *, const char*, const char*, ...)
  __attribute__ ((format (printf, 3, 0)));

int rdar6623513_aux(int len, const char* s) {
  rdar6623513(0, "hello", "%.*s", len, s);
}
