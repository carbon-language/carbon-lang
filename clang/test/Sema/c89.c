/* RUN: clang %s -std=c89 -pedantic -fsyntax-only -verify
 */
void test1() {
  {
    int i;
    i = i + 1;
    int j;          /* expected-warning {{mixing declarations and code}} */
  }
  {
    __extension__ int i;
    i = i + 1;
    int j;          /* expected-warning {{mixing declarations and code}} */
  }
  {
    int i;
    i = i + 1;
    __extension__ int j; /* expected-warning {{mixing declarations and code}} */
  }
}

long long test2;   /* expected-warning {{extension}} */


void test3(int i) {
  int A[i];        /* expected-warning {{variable length array}} */
}

int test4 = 0LL;		/* expected-warning {{long long}} */

/* PR1999 */
void test5(register);

/* PR2041 */
int *restrict;
int *__restrict;  /* expected-error {{expected identifier}} */


/* Implicit int, always ok */
test6() {}

/* PR2012 */
test7;  /* expected-warning {{declaration specifier missing, defaulting to 'int'}} */

void test8(int, x);  /* expected-warning {{declaration specifier missing, defaulting to 'int'}} */

typedef int sometype;
int a(sometype, y) {return 0;}  /* expected-warning {{declaration specifier missing, defaulting to 'int'}} */




void bar (void *); 
void f11 (z)       /* expected-error {{may not have 'void' type}} */
void z; 
{ bar (&z); }

typedef void T;
void foo(T); /* expected-warning {{empty parameter list defined with a typedef of 'void' is a C99 feature}} */

