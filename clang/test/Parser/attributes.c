// RUN: %clang_cc1 -fsyntax-only -verify %s -pedantic -std=c99

int __attribute__(()) x;

__inline void __attribute__((__always_inline__, __nodebug__))
foo(void) {
}


__attribute__(()) y;   // expected-warning {{defaults to 'int'}}

// PR2796
int (__attribute__(()) *z)(long y);


void f1(__attribute__(()) int x);

int f2(y, __attribute__(()) x);     // expected-error {{expected identifier}}

// This is parsed as a normal argument list (with two args that are implicit
// int) because the __attribute__ is a declspec.
void f3(__attribute__(()) x,  // expected-warning {{defaults to 'int'}}
        y);               // expected-warning {{defaults to 'int'}}

void f4(__attribute__(()));   // expected-error {{expected parameter declarator}}


// This is ok, the __attribute__ applies to the pointer.
int baz(int (__attribute__(()) *x)(long y));

void g1(void (*f1)(__attribute__(()) int x));
void g2(int (*f2)(y, __attribute__(()) x));    // expected-error {{expected identifier}}
void g3(void (*f3)(__attribute__(()) x, int y));  // expected-warning {{defaults to 'int'}}
void g4(void (*f4)(__attribute__(())));  // expected-error {{expected parameter declarator}}


void (*h1)(void (*f1)(__attribute__(()) int x));
void (*h2)(int (*f2)(y, __attribute__(()) x));    // expected-error {{expected identifier}}

void (*h3)(void (*f3)(__attribute__(()) x));   // expected-warning {{defaults to 'int'}}
void (*h4)(void (*f4)(__attribute__(())));  // expected-error {{expected parameter declarator}}



// rdar://6131260
int foo42(void) {
  int x, __attribute__((unused)) y, z;
  return 0;
}

// rdar://6096491
void __attribute__((noreturn)) d0(void), __attribute__((noreturn)) d1(void);

void d2(void) __attribute__((noreturn)), d3(void) __attribute__((noreturn));
