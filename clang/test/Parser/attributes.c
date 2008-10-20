// RUN: clang -fsyntax-only -verify %s -pedantic -std=c99

int __attribute__(()) x;  // expected-warning {{extension used}}

// Hide __attribute__ behind a macro, to silence extension warnings about
// "__attribute__ being an extension".
#define attribute __attribute__

__inline void attribute((__always_inline__, __nodebug__))
foo(void) {
}


attribute(()) y;   // expected-warning {{defaults to 'int'}}

// PR2796
int (attribute(()) *z)(long y);


void f1(attribute(()) int x);

int f2(y, attribute(()) x);     // expected-error {{expected identifier}}

// This is parsed as a normal argument list (with two args that are implicit
// int) because the attribute is a declspec.
void f3(attribute(()) x,  // expected-warning {{defaults to 'int'}}
        y);               // expected-warning {{defaults to 'int'}}

void f4(attribute(()));   // expected-error {{expected parameter declarator}}


// This is ok, the attribute applies to the pointer.
int baz(int (attribute(()) *x)(long y));

void g1(void (*f1)(attribute(()) int x));
void g2(int (*f2)(y, attribute(()) x));    // expected-error {{expected identifier}}
void g3(void (*f3)(attribute(()) x, int y));  // expected-warning {{defaults to 'int'}}
void g4(void (*f4)(attribute(())));  // expected-error {{expected parameter declarator}}


void (*h1)(void (*f1)(attribute(()) int x));
void (*h2)(int (*f2)(y, attribute(()) x));    // expected-error {{expected identifier}}

void (*h3)(void (*f3)(attribute(()) x));   // expected-warning {{defaults to 'int'}}
void (*h4)(void (*f4)(attribute(())));  // expected-error {{expected parameter declarator}}



// rdar://6131260
int foo42(void) {
  int x, attribute((unused)) y, z;
  return 0;
}

// rdar://6096491
void attribute((noreturn)) d0(void), attribute((noreturn)) d1(void);

