// RUN: clang-cc %s -fsyntax-only -verify -pedantic

extern int a1[];

void f0();
void f1(int [*]);
void f2(int [const *]);
void f3(int [volatile const*]);
int f4(*XX)(void); /* expected-error {{cannot return}} expected-warning {{type specifier missing, defaults to 'int'}} */

char ((((*X))));

void (*signal(int, void (*)(int)))(int);

int a, ***C, * const D, B(int);

int *A;

struct str;

int test2(int *P, int A) {
  struct str;

  // Hard case for array decl, not Array[*].
  int Array[*(int*)P+A];
}

typedef int atype;
int test3(x, 
          atype         /* expected-error {{unexpected type name 'atype': expected identifier}} */
         ) int x, atype; {}

int test4(x, x) int x; {} /* expected-error {{redefinition of parameter 'x'}} */


// PR3031
int (test5), ;  // expected-error {{expected identifier or '('}}



// PR3963 & rdar://6759604 - test error recovery for mistyped "typenames".

struct xyz { int y; };

foo_t a = 4;   // expected-error {{unknown type name 'foo_t'}}
xyz b;         // expected-error {{unknown type name 'xyz'}}

foo_t *d;      // expected-error {{unknown type name 'foo_t'}}

static f;      // expected-warning {{type specifier missing, defaults to 'int'}}
static g = 4;  // expected-warning {{type specifier missing, defaults to 'int'}}
static h        // expected-warning {{type specifier missing, defaults to 'int'}} 
      __asm__("foo"); // expected-warning {{extension used}}

int bar() { return a; }
